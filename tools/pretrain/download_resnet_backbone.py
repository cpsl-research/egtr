import argparse

import torch
from transformers import PretrainedConfig
from transformers.models.detr.modeling_detr import (
    DetrConvModel,
    DetrTimmConvEncoder,
    build_position_encoding,
)


def main(args):
    config = PretrainedConfig.from_pretrained("facebook/detr-resnet-50")
    backbone = DetrTimmConvEncoder(
        config.backbone, config.dilation, use_pretrained_backbone=True
    )
    position_embeddings = build_position_encoding(config)
    backbone = DetrConvModel(backbone, position_embeddings)
    torch.save(backbone.state_dict(), f"{args.backbone_dirpath}/resnet50.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backbone_dirpath", default="/data/shared/models/detr/backbone", type=str
    )
    args = parser.parse_args()
    main(args)
