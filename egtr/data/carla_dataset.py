import json
import os

import numpy as np
import torch
import torchvision
from tqdm import tqdm


class CarlaDetection(torchvision.datasets.CocoDetection):
    def __init__(self, data_folder, feature_extractor, split, debug=False):
        ann_file = os.path.join(data_folder, "annotations", f"{split}.json")
        img_folder = os.path.join(data_folder, "images", split)
        super(CarlaDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor
        self.split = split
        self.debug = debug

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CarlaDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target, "file_name": None}
        attrs = torch.Tensor([obj["attributes"] for obj in target["annotations"]])
        n_obj_before = len(target["annotations"])
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension
        target["class_labels"] -= 1  # remove 'no_relation' category
        n_obj_after = len(target["boxes"])

        # if the size happens to change during preprocess, (STILL DONT KNOW WHY)
        # we have to figure out which boxes remain....NOT SURE HOW TO DO THIS
        # therefore we will hack it by dropping relations/attributes from the end
        # HACK: THIS IS NOT CORRECT, but I'm assuming this only happens a few times...
        if n_obj_before != n_obj_after:
            # print("WARNING: THE NUMBER OF OBJECTS CHANGED DURING PREPROC!!!")
            # print("Dropping extras from the end of attrs, but this is not good...")
            attrs = attrs[:n_obj_after, :]

        # augment attributes
        target["attrs"] = attrs

        return pixel_values, target

    def __len__(self):
        if self.debug and self.split == "train":
            return 5000
        else:
            return len(self.ids)

    @staticmethod
    def unscale_attributes(attrs, attr_bounds):
        attr_bias_fct = attr_bounds[:, 0]
        attr_scale_fct = attr_bounds[:, 1] - attr_bounds[:, 0]
        attrs = attrs * attr_scale_fct[None, None, :] + attr_bias_fct[None, None, :]
        return attrs

    def get_image(self, idx):
        img, _ = super(CarlaDetection, self).__getitem__(idx)
        return img

    def get_target(self, idx):
        _, target = super(CarlaDetection, self).__getitem__(idx)
        return target


class CarlaDataset(CarlaDetection):
    def __init__(
        self, data_folder, feature_extractor, split, num_object_queries=100, debug=False
    ):
        super(CarlaDataset, self).__init__(data_folder, feature_extractor, split, debug)
        rel_file = os.path.join(data_folder, "relations", f"{split}_rel.json")
        with open(rel_file, "r") as f:
            rel = json.load(f)
        self.rel = rel["relations"]
        self.rel_categories = rel["rel_categories"][1:]  # remove 'no_relation' category
        self.num_relations = len(self.rel_categories)
        self.num_object_queries = num_object_queries

    def __getitem__(self, idx):
        # len(test_dataloader.dataset.coco.loadAnns(test_dataloader.dataset.coco.getAnnIds(1208)))
        # read in PIL image and target in COCO format
        img, target = super(CarlaDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target, "file_name": None}
        attrs = torch.Tensor([obj["attributes"] for obj in target["annotations"]])
        n_obj_before = len(target["annotations"])
        rel_list = self.rel[str(image_id)]
        rel = np.array(rel_list)
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension
        target["class_labels"] -= 1  # remove 'no_relation' category
        n_obj_after = len(target["boxes"])

        # if the size happens to change during preprocess, (STILL DONT KNOW WHY)
        # we have to figure out which boxes remain....NOT SURE HOW TO DO THIS
        # therefore we will hack it by dropping relations/attributes from the end
        # HACK: THIS IS NOT CORRECT, but I'm assuming this only happens a few times...
        if n_obj_before != n_obj_after:
            print("WARNING: THE NUMBER OF OBJECTS CHANGED DURING PREPROC!!!")
            print(
                "Dropping extras from the end of attrs/relations, but this is not good..."
            )
            attrs = attrs[:n_obj_after, :]
            rel = rel[(rel[:, 0] < n_obj_after) & (rel[:, 1] < n_obj_after), :]

        # augment relations etc.
        target["rel"] = self._get_rel_tensor(rel)
        target["attrs"] = attrs

        # sanity checks
        if len(attrs) != len(target["class_labels"]):
            raise RuntimeError(f"{len(attrs)} - {len(target['class_labels'])}")

        return pixel_values, target

    def get_relations(self, idx):
        return np.array(self.rel[str(self.ids[idx])])

    def _get_rel_tensor(self, rel_tensor):
        rel = torch.zeros(
            [self.num_object_queries, self.num_object_queries, self.num_relations]
        )
        indices = rel_tensor.T
        if len(indices) > 0:
            indices[-1, :] -= 1  # remove 'no_relation' category
            rel[indices[0, :], indices[1, :], indices[2, :]] = 1.0
        return rel


# https://github.com/suprosanna/relationformer/blob/75c24f61a81466df8f40c498e5f7aae3edd5ac6b/datasets/get_dataset_counts.py#L9
def carla_get_statistics(train_data, must_overlap=True):
    """
    Get counts of all of the relations. Used for modeling directly P(rel | o1, o2)
    :param train_data:
    :param must_overlap:
    :return:
    """
    num_classes = len(train_data.coco.cats)
    num_predicates = len(train_data.rel_categories)

    fg_matrix = np.zeros(
        (
            num_classes + 1,
            num_classes + 1,
            num_predicates,
        ),
        dtype=np.int64,
    )

    rel = train_data.rel
    for idx in tqdm(range(len(train_data))):
        image_id = train_data.ids[idx]

        target = train_data.coco.loadAnns(train_data.coco.getAnnIds(image_id))
        gt_classes = np.array(list(map(lambda x: x["category_id"], target)))
        rel_list = rel[str(image_id)]
        if len(rel_list) == 0:
            continue
        gt_indices = np.array(torch.Tensor(rel_list).T, dtype="int64")
        gt_indices[-1, :] -= 1

        # foreground
        o1o2 = gt_classes[gt_indices[:2, :]].T
        for (o1, o2), gtr in zip(o1o2, gt_indices[2]):
            fg_matrix[o1 - 1, o2 - 1, gtr] += 1

    return fg_matrix
