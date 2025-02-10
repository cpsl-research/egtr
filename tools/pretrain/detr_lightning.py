import pytorch_lightning as pl
import torch

from egtr.deformable_detr import DeformableDetrConfig, DeformableDetrForObjectDetection


class Detr(pl.LightningModule):
    def __init__(
        self,
        backbone_dirpath,
        auxiliary_loss,
        lr,
        lr_backbone,
        weight_decay,
        main_trained,
        id2label,
        num_queries,
        architecture,
        ce_loss_coefficient,
        coco_evaluator,
        oi_coco_evaluator,
        feature_extractor,
        train_dataloader,
        val_dataloader,
    ):
        super().__init__()
        # replace COCO classification head with custom head
        config = DeformableDetrConfig.from_pretrained(architecture)
        config.architecture = architecture
        config.auxiliary_loss = auxiliary_loss
        config.num_labels = max(id2label.keys()) + 1
        self._num_labels = max(id2label.keys()) + 1
        config.num_queries = num_queries
        config.ce_loss_coefficient = ce_loss_coefficient
        config.output_attention_states = False
        self.model = DeformableDetrForObjectDetection(config=config)
        self.model.model.backbone.load_state_dict(
            torch.load(f"{backbone_dirpath}/{config.backbone}.pt")
        )
        self.config = config

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.coco_evaluator = coco_evaluator
        self.oi_coco_evaluator = oi_coco_evaluator
        self.feature_extractor = feature_extractor
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        if main_trained:
            state_dict = torch.load(main_trained, map_location="cpu")["state_dict"]
            for k in list(state_dict.keys()):
                state_dict[k[6:]] = state_dict.pop(k)  # "model."
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
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

    def test_step(self, batch, batch_idx):
        # get the inputs
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        labels = [
            {k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]
        ]  # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack(
            [target["orig_size"] for target in labels], dim=0
        )
        results = self.feature_extractor.post_process(
            outputs,
            orig_target_sizes,
        )  # convert outputs of model to COCO api
        res = {
            target["image_id"].item(): output for target, output in zip(labels, results)
        }
        if self.coco_evaluator is not None:
            self.coco_evaluator.update(res)
        if self.oi_coco_evaluator is not None:
            self.oi_coco_evaluator(labels, res)

    def test_epoch_end(self, outputs):
        # log OD
        if self.coco_evaluator is not None:
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            self.log("AP50", self.coco_evaluator.coco_eval["bbox"].stats[1])
        if self.oi_coco_evaluator is not None:
            self.log_dict(self.oi_coco_evaluator.aggregate_metrics())

    def configure_optimizers(self):
        diff_lr_params = ["backbone", "reference_points", "sampling_offsets"]
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if (not any(nd in n for nd in diff_lr_params)) and p.requires_grad
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
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader
