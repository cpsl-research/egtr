# EGTR
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import json
import os
from glob import glob
from pathlib import Path

import torch
from egtr_lightning import SGG
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader

from egtr.data.carla_dataset import CarlaDataset, carla_get_statistics
from egtr.data.open_image import OIDataset, oi_get_statistics
from egtr.data.visual_genome import VGDataset, vg_get_statistics
from egtr.deformable_detr import (
    DeformableDetrFeatureExtractor,
    DeformableDetrFeatureExtractorWithAugmentorNoCrop,
)
from egtr.tools import collate_fn, get_checkpoint, str2bool
from lib.evaluation.coco_eval import CocoEvaluator
from lib.evaluation.oi_eval import OIEvaluator
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from util.misc import use_deterministic_algorithms


def main(args):
    # Feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=800, max_size=1333
    )
    feature_extractor_train = (
        DeformableDetrFeatureExtractorWithAugmentorNoCrop.from_pretrained(
            args.architecture, size=800, max_size=1333
        )
    )

    # Dataset
    if "visual_genome" in args.data_path:
        train_dataset = VGDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor_train,
            split="train",
            num_object_queries=args.num_queries,
            debug=args.debug,
        )
        val_dataset = VGDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split="val",
            num_object_queries=args.num_queries,
        )
        cats = train_dataset.coco.cats
        id2label = {k - 1: v["name"] for k, v in cats.items()}  # 0 ~ 149
        fg_matrix = vg_get_statistics(train_dataset, must_overlap=True)
    elif "carla" in args.data_path.lower():
        train_dataset = CarlaDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor_train,
            split="train",
            num_object_queries=args.num_queries,
            debug=args.debug,
        )
        val_dataset = CarlaDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split="val",
            num_object_queries=args.num_queries,
        )
        cats = train_dataset.coco.cats
        id2label = {k - 1: v["name"] for k, v in cats.items()}
        fg_matrix = carla_get_statistics(train_dataset, must_overlap=True)
    else:
        train_dataset = OIDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor_train,
            split="train",
            filter_duplicate_rels=args.filter_duplicate_rels,
            filter_multiple_rels=args.filter_multiple_rels,
            num_object_queries=args.num_queries,
            debug=args.debug,
        )
        val_dataset = OIDataset(
            data_folder=args.data_path,
            split="val",
            num_object_queries=args.num_queries,
            feature_extractor=feature_extractor,
        )
        id2label = train_dataset.classes_to_ind  # 0 ~ 600
        fg_matrix = oi_get_statistics(train_dataset, must_overlap=True)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # Evaluator
    rel_categories = train_dataset.rel_categories
    multiple_sgg_evaluator = None
    single_sgg_evaluator = None
    coco_evaluator = None
    oi_evaluator = None

    multiple_sgg_evaluator_list = []
    single_sgg_evaluator_list = []
    if args.eval_when_train_end:
        if args.eval_multiple_preds:
            multiple_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(
                multiple_preds=True
            )  # R@k
            for index, name in enumerate(rel_categories):
                multiple_sgg_evaluator_list.append(
                    (
                        index,
                        name,
                        BasicSceneGraphEvaluator.all_modes(multiple_preds=True),
                    )
                )
        if args.eval_single_preds:
            single_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(
                multiple_preds=False
            )  # R@k
            for index, name in enumerate(rel_categories):
                single_sgg_evaluator_list.append(
                    (
                        index,
                        name,
                        BasicSceneGraphEvaluator.all_modes(multiple_preds=False),
                    )
                )
        if "visual_genome" in args.data_path:
            coco_evaluator = CocoEvaluator(
                val_dataset.coco, ["bbox"]
            )  # initialize evaluator with ground truths
        elif "open-image" in args.data_path:
            oi_evaluator = OIEvaluator(
                train_dataset.rel_categories, train_dataset.ind_to_classes
            )

    # Logger setting
    save_dir = f"{args.output_path}/egtr__{'/'.join(args.pretrained.split('/')[-3:]).replace('/', '__')}"
    if args.from_scratch:
        save_dir += "__from_scratch"
    name = f"batch__{args.batch_size * args.gpus * args.accumulate}__epochs__{args.max_epochs}_{args.max_epochs_finetune}__lr__{args.lr_backbone}_{args.lr}_{args.lr_initialized}"
    if args.memo:
        name += f"__{args.memo}"
    if args.debug:
        name += "__debug"
    if args.resume:
        version = args.version  # for resuming
    else:
        version = None  #  If version is not specified the logger inspects the save directory for existing versions, then automatically assigns the next available version.

    # Trainer setting
    logger = TensorBoardLogger(save_dir, name=name, version=version)

    # load a set checkpoint, if available and NOT resume
    ckpt_path = get_checkpoint(args.resume, args.initial_ckpt_dir, logger.log_dir)

    # Module
    module = SGG(
        architecture=args.architecture,
        backbone_dirpath=args.backbone_dirpath,
        auxiliary_loss=args.auxiliary_loss,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        lr_initialized=args.lr_initialized,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
        main_trained="",
        from_scratch=args.from_scratch,
        id2label=id2label,
        rel_loss_coefficient=args.rel_loss_coefficient,
        smoothing=args.smoothing,
        rel_sample_negatives=args.rel_sample_negatives,
        rel_sample_nonmatching=args.rel_sample_nonmatching,
        rel_categories=rel_categories,
        multiple_sgg_evaluator=multiple_sgg_evaluator,
        multiple_sgg_evaluator_list=multiple_sgg_evaluator_list,
        single_sgg_evaluator=single_sgg_evaluator,
        single_sgg_evaluator_list=single_sgg_evaluator_list,
        coco_evaluator=coco_evaluator,
        oi_evaluator=oi_evaluator,
        feature_extractor=feature_extractor,
        num_queries=args.num_queries,
        ce_loss_coefficient=args.ce_loss_coefficient,
        rel_sample_negatives_largest=args.rel_sample_negatives_largest,
        rel_sample_nonmatching_largest=args.rel_sample_nonmatching_largest,
        use_freq_bias=args.use_freq_bias,
        fg_matrix=fg_matrix,
        use_log_softmax=args.use_log_softmax,
        freq_bias_eps=1e-12,
        connectivity_loss_coefficient=args.connectivity_loss_coefficient,
        logit_adjustment=args.logit_adjustment,
        logit_adj_tau=args.logit_adj_tau,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    # Callback
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        filename="{epoch:02d}-{validation_loss:.2f}",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss", patience=args.patience, verbose=True, mode="min"
    )

    # Train
    trainer = None
    if not args.skip_train:
        # Main training
        if not Path(
            TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            ).log_dir
        ).exists():
            # Training
            trainer = Trainer(
                precision=args.precision,
                logger=logger,
                gpus=args.gpus,
                max_epochs=args.max_epochs,
                gradient_clip_val=args.gradient_clip_val,
                strategy=DDPStrategy(find_unused_parameters=False),
                callbacks=[checkpoint_callback, early_stop_callback],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Main training")
            trainer.fit(module, ckpt_path=ckpt_path)

            try:
                os.chmod(logger.log_dir, 0o0777)
            except PermissionError as e:
                print(e)

        if args.finetune:
            ckpt_path = sorted(  # load best model
                glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
                key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
            )[-1]

            # Finetune trainer setting
            logger = TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            )
            if os.path.exists(f"{logger.log_dir}/checkpoints"):
                finetune_ckpt_path = f"{logger.log_dir}/checkpoints/last.ckpt"
            else:
                finetune_ckpt_path = None

            # Finetune module
            module = SGG(
                architecture=args.architecture,
                backbone_dirpath=args.backbone_dirpath,
                auxiliary_loss=args.auxiliary_loss,
                lr=args.lr * 0.1,
                lr_backbone=args.lr_backbone * 0.1,
                lr_initialized=args.lr_initialized * 0.1,
                weight_decay=args.weight_decay,
                pretrained=args.pretrained,
                main_trained=ckpt_path,
                from_scratch=args.from_scratch,
                id2label=id2label,
                rel_loss_coefficient=args.rel_loss_coefficient,
                smoothing=args.smoothing,
                rel_sample_negatives=args.rel_sample_negatives,
                rel_sample_nonmatching=args.rel_sample_nonmatching,
                rel_categories=rel_categories,
                multiple_sgg_evaluator=multiple_sgg_evaluator,
                multiple_sgg_evaluator_list=multiple_sgg_evaluator_list,
                single_sgg_evaluator=single_sgg_evaluator,
                single_sgg_evaluator_list=single_sgg_evaluator_list,
                coco_evaluator=coco_evaluator,
                oi_evaluator=oi_evaluator,
                feature_extractor=feature_extractor,
                num_queries=args.num_queries,
                ce_loss_coefficient=args.ce_loss_coefficient,
                rel_sample_negatives_largest=args.rel_sample_negatives_largest,
                rel_sample_nonmatching_largest=args.rel_sample_nonmatching_largest,
                use_freq_bias=args.use_freq_bias,
                fg_matrix=fg_matrix,
                use_log_softmax=args.use_log_softmax,
                freq_bias_eps=1e-12,
                connectivity_loss_coefficient=args.connectivity_loss_coefficient,
                logit_adjustment=args.logit_adjustment,
                logit_adj_tau=args.logit_adj_tau,
            )

            # Finetune callback
            checkpoint_callback = ModelCheckpoint(
                monitor="validation_loss",
                filename="{epoch:02d}-{validation_loss:.2f}",
                save_last=True,
            )
            early_stop_callback = EarlyStopping(
                monitor="validation_loss",
                patience=args.patience,
                verbose=True,
                mode="min",
            )

            # Training
            trainer = Trainer(
                precision=args.precision,
                logger=logger,
                max_epochs=args.max_epochs_finetune,
                gpus=args.gpus,
                gradient_clip_val=args.gradient_clip_val,
                strategy=DDPStrategy(find_unused_parameters=False),
                callbacks=[checkpoint_callback, early_stop_callback],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Finetune with smaller lr")
            trainer.fit(module, ckpt_path=finetune_ckpt_path)

        if trainer is not None:
            torch.distributed.destroy_process_group()
            try:
                os.chmod(logger.log_dir, 0o0777)
            except PermissionError as e:
                print(e)

    # Evaluation
    if args.eval_when_train_end and (trainer is None or trainer.is_global_zero):
        if args.skip_train and args.finetune:
            logger = TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            )

        # Load best model
        ckpt_path = sorted(
            glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
            key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
        )[-1]
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        module.model.load_state_dict(state_dict)  # load best model

        # Eval
        trainer = Trainer(
            precision=args.precision, logger=logger, gpus=1, max_epochs=-1
        )
        if "visual_genome" in args.data_path:
            Dataset = VGDataset
        elif "carla" in args.data_path:
            Dataset = CarlaDataset
        else:
            Dataset = OIDataset
        test_dataset = Dataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
            num_object_queries=args.num_queries,
        )
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=lambda x: collate_fn(x, feature_extractor),
            batch_size=args.eval_batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
            persistent_workers=True,
        )
        if trainer.is_global_zero:
            print("### Evaluation")
        metric = trainer.test(module, dataloaders=test_dataloader)

        # Save eval metric
        metric = metric[0]
        device = "".join(torch.cuda.get_device_name(0).split()[1:2])
        filename = f'{ckpt_path.replace(".ckpt", "")}__{args.split}__{len(test_dataloader)}__{device}'
        if args.logit_adjustment:
            filename += f"__la_{args.logit_adj_tau}"
        metric["eval_arg"] = args.__dict__
        with open(f"{filename}.json", "w") as f:
            json.dump(metric, f)
        print("metric is saved in", f"{filename}.json")


if __name__ == "__main__":

    seed_everything(42, workers=True)

    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument("--data_path", type=str, default="dataset/visual_genome")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--backbone_dirpath", type=str, default=""
    )  # required when from_scratch is True
    parser.add_argument("--initial_ckpt_dir", type=str, required=False)

    # Architecture
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--auxiliary_loss", type=str2bool, default=False)
    parser.add_argument(
        "--from_scratch", type=str2bool, default=False
    )  # whether to train without pretrained detr
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
    )  # set to "architecture" when from_scratch is True

    # Hyperparameters
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--ce_loss_coefficient", type=float, default=2.0)
    parser.add_argument("--rel_loss_coefficient", type=float, default=15.0)
    parser.add_argument(
        "--connectivity_loss_coefficient", type=float, default=30.0
    )  # OI: 90
    parser.add_argument("--smoothing", type=float, default=1e-14)
    parser.add_argument("--rel_sample_negatives", type=int, default=80)
    parser.add_argument("--rel_sample_nonmatching", type=int, default=80)
    parser.add_argument(
        "--rel_sample_negatives_largest", type=str2bool, default=True
    )  # OI: True
    parser.add_argument(
        "--rel_sample_nonmatching_largest", type=str2bool, default=True
    )  # OI: False

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--max_epochs_finetune", type=int, default=25)
    parser.add_argument("--lr_backbone", type=float, default=2e-7)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--lr_initialized", type=float, default=2e-4)  # for pretrained
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)

    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--memo", type=str, default="")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--finetune", type=str2bool, default=True)

    parser.add_argument(
        "--filter_duplicate_rels", type=str2bool, default=True
    )  # for OI
    parser.add_argument("--filter_multiple_rels", type=str2bool, default=True)  # for OI
    parser.add_argument("--use_freq_bias", type=str2bool, default=True)
    parser.add_argument("--use_log_softmax", type=str2bool, default=False)

    # Evaluation
    parser.add_argument("--skip_train", type=str2bool, default=False)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_when_train_end", type=str2bool, default=True)
    parser.add_argument("--eval_single_preds", type=str2bool, default=True)
    parser.add_argument("--eval_multiple_preds", type=str2bool, default=False)

    parser.add_argument("--logit_adjustment", type=str2bool, default=False)
    parser.add_argument("--logit_adj_tau", type=float, default=0.3)

    # Speed up
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])

    args = parser.parse_args()
    if args.from_scratch:
        args.pretrained = args.architecture

    main(args)
