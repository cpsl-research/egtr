import os
from argparse import ArgumentTypeError
from glob import glob


def collate_fn(batch, feature_extractor):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt"
    )
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def get_last_ckpt(ckpt_dir):
    if os.path.exists(f"{ckpt_dir}/last.ckpt"):
        ckpt_path = f"{ckpt_dir}/last.ckpt"
    else:
        ckpt_path = sorted(
            glob(f"{ckpt_dir}/epoch=*.ckpt"),
            key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
        )[-1]
    return ckpt_path


def get_checkpoint(resume: bool, initial_ckpt_dir: str, log_dir: str):
    # load a set checkpoint, if available and NOT resume
    if (not resume) and initial_ckpt_dir:
        ckpt_dir = f"{initial_ckpt_dir}/checkpoints"
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError(ckpt_dir)
        else:
            ckpt_path = get_last_ckpt(ckpt_dir)
    else:
        ckpt_dir = f"{log_dir}/checkpoints"
        if os.path.exists(ckpt_dir):
            ckpt_path = get_last_ckpt(ckpt_dir)
        else:
            ckpt_path = None
    return ckpt_path
