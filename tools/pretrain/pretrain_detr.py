# Reference: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb

import argparse
import os
from pathlib import Path

import torch
from detr_lightning import Detr
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader

from egtr.data.carla_dataset import CarlaDetection
from egtr.data.open_image import OIDetection
from egtr.data.visual_genome import VGDetection
from egtr.deformable_detr import (
    DeformableDetrFeatureExtractor,
    DeformableDetrFeatureExtractorWithAugmentor,
)
from egtr.tools import collate_fn, get_checkpoint, str2bool
from lib.evaluation.coco_eval import CocoEvaluator
from lib.evaluation.oi_eval import OICocoEvaluator
from util.misc import use_deterministic_algorithms


def get_trainval_dataloaders(args, feature_extractor):
    # Dataset
    if "visual_genome" in args.data_path:
        train_dataset = VGDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split="train",
            debug=args.debug,
        )
        val_dataset = VGDetection(
            data_folder=args.data_path, feature_extractor=feature_extractor, split="val"
        )
        cats = train_dataset.coco.cats
        id2label = {k - 1: v["name"] for k, v in cats.items()}  # 0 ~ 149
    elif "carla" in args.data_path.lower():
        train_dataset = CarlaDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split="train",
            debug=args.debug,
        )
        val_dataset = CarlaDetection(
            data_folder=args.data_path, feature_extractor=feature_extractor, split="val"
        )
        cats = train_dataset.coco.cats
        id2label = {k - 1: v["name"] for k, v in cats.items()}
    else:
        train_dataset = OIDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split="train",
            debug=args.debug,
        )
        val_dataset = OIDetection(
            data_folder=args.data_path, feature_extractor=feature_extractor, split="val"
        )
        id2label = train_dataset.classes_to_ind  # 0 ~ 600
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

    return train_dataset, val_dataset, train_dataloader, val_dataloader, id2label


def get_test_dataloader(args, feature_extractor):
    if "visual_genome" in args.data_path:
        test_dataset = VGDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
        )
    elif "carla" in args.data_path.lower():
        test_dataset = CarlaDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
        )
    else:
        test_dataset = OIDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
        )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.eval_batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    return test_dataloader


def main(args):
    # Feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=800, max_size=1333
    )
    feature_extractor_train = (
        DeformableDetrFeatureExtractorWithAugmentor.from_pretrained(
            args.architecture, size=800, max_size=1333
        )
    )
    (
        train_dataset,
        val_dataset,
        train_dataloader,
        val_dataloader,
        id2label,
    ) = get_trainval_dataloaders(args, feature_extractor_train)

    # Evaluator
    if args.eval_when_train_end:
        if ("visual_genome" in args.data_path) or ("carla" in args.data_path.lower()):
            coco_evaluator = CocoEvaluator(
                val_dataset.coco, ["bbox"]
            )  # initialize evaluator with ground truths
            oi_coco_evaluator = None
        elif "open-image" in args.data_path:
            oi_coco_evaluator = OICocoEvaluator(
                train_dataset.rel_categories, train_dataset.ind_to_classes
            )
            coco_evaluator = None
        else:
            raise NotImplementedError(args.data_path)
    else:
        coco_evaluator = None
        oi_coco_evaluator = None

    # Logger setting
    save_dir = (
        f"{args.output_path}/pretrained_detr__{args.architecture.replace('/', '__')}"
    )
    name = f"batch__{args.batch_size * args.gpus * args.accumulate}__epochs__{args.max_epochs}_{args.max_epochs_finetune}__lr__{args.lr_backbone}_{args.lr}"
    if args.memo:
        name += f"__{args.memo}"
    if args.debug:
        name += "__debug"
    if args.resume:
        version = args.version  # for resuming
    else:
        version = None  #  If version is not specified the logger inspects the save directory for existing versions, then automatically assigns the next available version.

    # Trainer setting
    # TODO: with multiple gpus, ther versioning folder save gets screwed up
    logger = TensorBoardLogger(save_dir, name=name, version=version)

    # load a set checkpoint, if available and NOT resume
    ckpt_path = get_checkpoint(args.resume, args.initial_ckpt_dir, logger.log_dir)

    # Module
    module = Detr(
        backbone_dirpath=args.backbone_dirpath,
        auxiliary_loss=args.auxiliary_loss,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        main_trained="",
        id2label=id2label,
        num_queries=args.num_queries,
        architecture=args.architecture,
        ce_loss_coefficient=args.ce_loss_coefficient,
        coco_evaluator=coco_evaluator,
        oi_coco_evaluator=oi_coco_evaluator,
        feature_extractor=feature_extractor,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    # Save config
    module.config.save_pretrained(logger.log_dir)

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

        # Finetuning
        if args.finetune:
            ckpt_path = get_checkpoint(
                args.resume, args.initial_ckpt_dir, logger.log_dir
            )

            # Finetune trainer setting
            logger = TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            )
            if os.path.exists(f"{logger.log_dir}/checkpoints"):
                finetune_ckpt_path = f"{logger.log_dir}/checkpoints/last.ckpt"
            else:
                finetune_ckpt_path = None

            # Finetune module
            module = Detr(
                backbone_dirpath=args.backbone_dirpath,
                auxiliary_loss=args.auxiliary_loss,
                lr=args.lr * 0.1,
                lr_backbone=args.lr_backbone * 0.1,
                weight_decay=args.weight_decay,
                main_trained=ckpt_path,
                id2label=id2label,
                num_queries=args.num_queries,
                architecture=args.architecture,
                ce_loss_coefficient=args.ce_loss_coefficient,
                coco_evaluator=coco_evaluator,
                oi_coco_evaluator=oi_coco_evaluator,
                feature_extractor=feature_extractor,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
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
                gpus=args.gpus,
                max_epochs=args.max_epochs_finetune,
                gradient_clip_val=args.gradient_clip_val,
                strategy=DDPStrategy(find_unused_parameters=False),
                callbacks=[checkpoint_callback, early_stop_callback],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Finetune with smaller lr")
            trainer.fit(module, ckpt_path=finetune_ckpt_path)

        # load best model & save best model as pytorch_model.bin
        ckpt_path = get_checkpoint(args.resume, args.initial_ckpt_dir, logger.log_dir)
        # ckpt_path = sorted(
        #     glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
        #     key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
        # )[-1]
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        module.model.load_state_dict(state_dict)
        if trainer.is_global_zero:
            module.model.save_pretrained(logger.log_dir)

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
        ckpt_path = get_checkpoint(args.resume, args.initial_ckpt_dir, logger.log_dir)
        # ckpt_path = sorted(
        #     glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
        #     key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
        # )[-1]
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        module.model.load_state_dict(state_dict)

        # Eval
        trainer = Trainer(
            precision=args.precision, logger=logger, gpus=1, max_epochs=-1
        )
        test_dataloader = get_test_dataloader(args, feature_extractor)
        if trainer.is_global_zero:
            print("### Evaluation")
        trainer.test(module, dataloaders=test_dataloader)


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
    parser.add_argument("--backbone_dirpath", type=str, required=True)
    parser.add_argument("--initial_ckpt_dir", type=str, required=False)
    parser.add_argument("--load_initial_ckpt", type=str2bool, default=False)

    # Architecture
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--auxiliary_loss", type=str2bool, default=True)

    # Hyperparameters
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--ce_loss_coefficient", type=float, default=2.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--max_epochs_finetune", type=int, default=50)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)

    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--memo", type=str, default="")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--finetune", type=str2bool, default=True)

    # Evaluation
    parser.add_argument("--skip_train", type=str2bool, default=False)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_when_train_end", type=str2bool, default=True)

    # Speed up
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])

    args = parser.parse_args()
    main(args)
