import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from egtr.deformable_detr import DeformableDetrConfig
from egtr.egtr import DetrForSceneGraphGeneration
from lib.evaluation.sg_eval import calculate_mR_from_evaluator_list
from lib.pytorch_misc import argsort_desc
from util.box_ops import rescale_bboxes


class SGG(pl.LightningModule):
    def __init__(
        self,
        architecture,
        backbone_dirpath,
        auxiliary_loss,
        lr,
        lr_backbone,
        lr_initialized,
        weight_decay,
        pretrained,
        main_trained,
        from_scratch,
        id2label,
        rel_loss_coefficient,
        smoothing,
        rel_sample_negatives,
        rel_sample_nonmatching,
        rel_categories,
        multiple_sgg_evaluator,
        multiple_sgg_evaluator_list,
        single_sgg_evaluator,
        single_sgg_evaluator_list,
        coco_evaluator,
        oi_evaluator,
        feature_extractor,
        num_queries,
        ce_loss_coefficient,
        rel_sample_negatives_largest,
        rel_sample_nonmatching_largest,
        use_freq_bias,
        fg_matrix,
        use_log_softmax,
        freq_bias_eps,
        connectivity_loss_coefficient,
        logit_adjustment,
        logit_adj_tau,
        train_dataloader,
        val_dataloader,
    ):

        super().__init__()
        # replace COCO classification head with custom head
        config = DeformableDetrConfig.from_pretrained(pretrained)
        config.architecture = architecture
        config.auxiliary_loss = auxiliary_loss
        config.from_scratch = from_scratch
        config.num_rel_labels = len(rel_categories)
        config.num_labels = max(id2label.keys()) + 1
        config.num_queries = num_queries
        config.rel_loss_coefficient = rel_loss_coefficient
        config.smoothing = smoothing
        config.rel_sample_negatives = rel_sample_negatives
        config.rel_sample_nonmatching = rel_sample_nonmatching
        config.ce_loss_coefficient = ce_loss_coefficient
        config.pretrained = pretrained
        config.rel_sample_negatives_largest = rel_sample_negatives_largest
        config.rel_sample_nonmatching_largest = rel_sample_nonmatching_largest

        config.connectivity_loss_coefficient = connectivity_loss_coefficient
        config.use_freq_bias = use_freq_bias
        config.use_log_softmax = use_log_softmax
        config.freq_bias_eps = freq_bias_eps

        config.logit_adjustment = logit_adjustment
        config.logit_adj_tau = logit_adj_tau
        self.config = config

        if config.from_scratch:
            assert backbone_dirpath
            self.model = DetrForSceneGraphGeneration(config=config, fg_matrix=fg_matrix)
            self.model.model.backbone.load_state_dict(
                torch.load(f"{backbone_dirpath}/{config.backbone}.pt")
            )
            self.initialized_keys = []
        else:
            self.model, load_info = DetrForSceneGraphGeneration.from_pretrained(
                pretrained,
                config=config,
                ignore_mismatched_sizes=True,
                output_loading_info=True,
                fg_matrix=fg_matrix,
            )
            self.initialized_keys = load_info["missing_keys"] + [
                _key for _key, _, _ in load_info["mismatched_keys"]
            ]

        if main_trained:
            state_dict = torch.load(main_trained, map_location="cpu")["state_dict"]
            for k in list(state_dict.keys()):
                state_dict[k[6:]] = state_dict.pop(k)  # "model."
            self.model.load_state_dict(state_dict, strict=False)

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.lr_initialized = lr_initialized
        self.weight_decay = weight_decay
        self.multiple_sgg_evaluator = multiple_sgg_evaluator
        self.multiple_sgg_evaluator_list = multiple_sgg_evaluator_list
        self.single_sgg_evaluator = single_sgg_evaluator
        self.single_sgg_evaluator_list = single_sgg_evaluator_list
        self.coco_evaluator = coco_evaluator
        self.oi_evaluator = oi_evaluator
        self.feature_extractor = feature_extractor
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        del outputs
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "training_loss": loss.item(),
        }
        log_dict.update({f"training_{k}": v.item() for k, v in loss_dict.items()})
        self.log_dict(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        loss_dict["loss"] = loss
        del loss
        return loss_dict

    def validation_epoch_end(self, outputs):
        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "epoch": torch.tensor(self.current_epoch, dtype=torch.float32),
        }
        for k in outputs[0].keys():
            log_dict[f"validation_" + k] = (
                torch.stack([x[k] for x in outputs]).mean().item()
            )
        self.log_dict(log_dict, on_epoch=True)

    @rank_zero_only
    def on_train_start(self) -> None:
        self.config.save_pretrained(self.logger.log_dir)
        return super().on_train_start()

    def test_step(self, batch, batch_idx):
        # get the inputs
        self.model.eval()

        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        targets = [{k: v.cpu() for k, v in label.items()} for label in batch["labels"]]

        with torch.no_grad():
            outputs = self.forward(pixel_values, pixel_mask)
            # eval SGG
            evaluate_batch(
                outputs,
                targets,
                self.multiple_sgg_evaluator,
                self.multiple_sgg_evaluator_list,
                self.single_sgg_evaluator,
                self.single_sgg_evaluator_list,
                self.oi_evaluator,
                self.config.num_labels,
            )
            # eval OD
            if self.coco_evaluator is not None:
                orig_target_sizes = torch.stack(
                    [target["orig_size"] for target in targets], dim=0
                )
                results = self.feature_extractor.post_process(
                    outputs, orig_target_sizes.to(self.device)
                )  # convert outputs of model to COCO api
                res = {
                    target["image_id"].item(): output
                    for target, output in zip(targets, results)
                }
                self.coco_evaluator.update(res)

    def test_epoch_end(self, outputs):
        log_dict = {}
        # log OD
        if self.coco_evaluator is not None:
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            log_dict.update({"AP50": self.coco_evaluator.coco_eval["bbox"].stats[1]})

        # log SGG
        if self.multiple_sgg_evaluator is not None:
            recall = self.multiple_sgg_evaluator["sgdet"].print_stats()
            mean_recall = calculate_mR_from_evaluator_list(
                self.multiple_sgg_evaluator_list, "sgdet", multiple_preds=True
            )
            log_dict.update(recall)
            log_dict.update(mean_recall)

        if self.single_sgg_evaluator is not None:
            recall = self.single_sgg_evaluator["sgdet"].print_stats()
            mean_recall = calculate_mR_from_evaluator_list(
                self.single_sgg_evaluator_list, "sgdet", multiple_preds=False
            )
            recall = dict(zip(["(single)" + x for x in recall.keys()], recall.values()))
            mean_recall = dict(
                zip(["(single)" + x for x in mean_recall.keys()], mean_recall.values())
            )
            log_dict.update(recall)
            log_dict.update(mean_recall)

        if self.oi_evaluator is not None:
            metrics = self.oi_evaluator.aggregate_metrics()
            log_dict.update(metrics)
        self.log_dict(log_dict, on_epoch=True)
        return log_dict

    def configure_optimizers(self):
        diff_lr_params = ["backbone", "reference_points", "sampling_offsets"]

        if self.lr_initialized is not None:  # rel_predictor
            initialized_lr_params = self.initialized_keys
        else:
            initialized_lr_params = []
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if (not any(nd in n for nd in diff_lr_params))
                    and (not any(nd in n for nd in initialized_lr_params))
                    and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in diff_lr_params) and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        if initialized_lr_params:
            param_dicts.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in initialized_lr_params)
                        and p.requires_grad
                    ],
                    "lr": self.lr_initialized,
                }
            )
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader


# Reference: https://github.com/yuweihao/KERN/blob/master/models/eval_rels.py
def evaluate_batch(
    outputs,
    targets,
    multiple_sgg_evaluator,
    multiple_sgg_evaluator_list,
    single_sgg_evaluator,
    single_sgg_evaluator_list,
    oi_evaluator,
    num_labels,
    max_topk=100,
):
    for j, target in enumerate(targets):
        # Pred
        pred_logits = outputs["logits"][j]
        obj_scores, pred_classes = torch.max(
            pred_logits.softmax(-1)[:, :num_labels], -1
        )
        sub_ob_scores = torch.outer(obj_scores, obj_scores)
        sub_ob_scores[
            torch.arange(pred_logits.size(0)), torch.arange(pred_logits.size(0))
        ] = 0.0  # prevent self-connection

        pred_boxes = outputs["pred_boxes"][j]
        pred_rel = torch.clamp(outputs["pred_rel"][j], 0.0, 1.0)
        if "pred_connectivity" in outputs:
            pred_connectivity = torch.clamp(outputs["pred_connectivity"][j], 0.0, 1.0)
            pred_rel = torch.mul(pred_rel, pred_connectivity)

        # GT
        orig_size = target["orig_size"]
        target_labels = target["class_labels"]  # [num_objs]
        target_boxes = target["boxes"]  # [num_objs, 4]
        target_rel = target["rel"].nonzero()  # [num_rels, 3(s, o, p)]

        gt_entry = {
            "gt_relations": target_rel.clone().numpy(),
            "gt_boxes": rescale_bboxes(target_boxes, torch.flip(orig_size, dims=[0]))
            .clone()
            .numpy(),
            "gt_classes": target_labels.clone().numpy(),
        }

        if multiple_sgg_evaluator is not None:
            triplet_scores = torch.mul(pred_rel, sub_ob_scores.unsqueeze(-1))
            pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().numpy())[
                :max_topk, :
            ]  # [pred_rels, 3(s,o,p)]
            rel_scores = (
                pred_rel.cpu()
                .clone()
                .numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1], pred_rel_inds[:, 2]]
            )  # [pred_rels]

            pred_entry = {
                "pred_boxes": rescale_bboxes(
                    pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
                )
                .clone()
                .numpy(),
                "pred_classes": pred_classes.cpu().clone().numpy(),
                "obj_scores": obj_scores.cpu().clone().numpy(),
                "pred_rel_inds": pred_rel_inds,
                "rel_scores": rel_scores,
            }
            multiple_sgg_evaluator["sgdet"].evaluate_scene_graph_entry(
                gt_entry, pred_entry
            )

            for pred_id, _, evaluator_rel in multiple_sgg_evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel["gt_relations"][:, -1], pred_id)
                gt_entry_rel["gt_relations"] = gt_entry_rel["gt_relations"][mask, :]
                if gt_entry_rel["gt_relations"].shape[0] == 0:
                    continue
                evaluator_rel["sgdet"].evaluate_scene_graph_entry(
                    gt_entry_rel, pred_entry
                )

        if single_sgg_evaluator is not None:
            triplet_scores = torch.mul(pred_rel.max(-1)[0], sub_ob_scores)
            pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().numpy())[
                :max_topk, :
            ]  # [pred_rels, 2(s,o)]
            rel_scores = (
                pred_rel.cpu().clone().numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1]]
            )  # [pred_rels, 50]

            pred_entry = {
                "pred_boxes": rescale_bboxes(
                    pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
                )
                .clone()
                .numpy(),
                "pred_classes": pred_classes.cpu().clone().numpy(),
                "obj_scores": obj_scores.cpu().clone().numpy(),
                "pred_rel_inds": pred_rel_inds,
                "rel_scores": rel_scores,
            }
            single_sgg_evaluator["sgdet"].evaluate_scene_graph_entry(
                gt_entry, pred_entry
            )
            for pred_id, _, evaluator_rel in single_sgg_evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel["gt_relations"][:, -1], pred_id)
                gt_entry_rel["gt_relations"] = gt_entry_rel["gt_relations"][mask, :]
                if gt_entry_rel["gt_relations"].shape[0] == 0:
                    continue
                evaluator_rel["sgdet"].evaluate_scene_graph_entry(
                    gt_entry_rel, pred_entry
                )

        if oi_evaluator is not None:  # OI evaluation, return all possible indicies
            sbj_obj_inds = torch.cartesian_prod(
                torch.arange(pred_logits.shape[0]), torch.arange(pred_logits.shape[0])
            )
            pred_scores = (
                pred_rel.cpu().clone().numpy().reshape(-1, pred_rel.size(-1))
            )  # (num_obj * num_obj, num_rel_classes)

            pred_entry = {
                "pred_boxes": rescale_bboxes(
                    pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
                )
                .clone()
                .numpy(),
                "pred_classes": pred_classes.cpu().clone().numpy(),
                "obj_scores": obj_scores.cpu().clone().numpy(),
                "sbj_obj_inds": sbj_obj_inds,  # for oi, (num_obj * num_obj, num_rel_classes)
                "pred_scores": pred_scores,  # for oi, (num_obj * num_obj, num_rel_classes)
            }
            oi_evaluator(gt_entry, pred_entry)
