import argparse
import json
import os
import shutil

import numpy as np
from avapi.carla import CarlaScenesManager
from avstack.geometry import ReferenceFrame, q_cam_to_stan
from tqdm import tqdm

from egtr.relations import REL_REVINDEX, REL_STRINGS, RELATIONS


def main(args):
    """Convert Carla dataset to the COCO format"""

    # make dir structure
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    img_dir = os.path.join(args.output_dir, "images")
    ann_dir = os.path.join(args.output_dir, "annotations")
    os.makedirs(img_dir)
    os.makedirs(ann_dir)
    if args.with_relations:
        rel_dir = os.path.join(args.output_dir, "relations")
        os.makedirs(rel_dir)

    # set up categories
    category_ids = {
        "person": 1,
        "car": 2,
        "motorcycle": 3,
        "bicycle": 4,
        "truck": 5,
    }
    categories = [{"id": v, "name": k} for k, v in category_ids.items()]

    # set up attribute scaling
    attr_bounds = {
        "range_obj": [0, 100],
        "volume_3d": [0, 100],
        "fraction_visible": [0, 1],
        "orientation_3d": [-np.pi, np.pi],
    }
    attributes = [{"name": k, "bound": v} for k, v in attr_bounds.items()]

    # set up relationships
    relations = {}

    # loop over the splits
    idx_img = 0
    idx_ann = 0
    CSM = CarlaScenesManager(args.input_dir)
    for split, scenes in CSM.splits_scenes.items():
        os.makedirs(os.path.join(img_dir, split))
        images = []
        annotations = []

        # loop over scenes
        for idx_scene, scene in enumerate(scenes):
            # get dataset manager
            CDM = CSM.get_scene_dataset_by_name(scene)

            # loop over agents and cameras
            agents = CDM.agent_IDs
            for idx_agent, agent in enumerate(agents):
                for sensor in CDM.sensor_IDs[agent]:
                    if "camera" in sensor:
                        print(
                            f"Processing scene {idx_scene+1}/{len(scenes)}, agent {idx_agent+1}/{len(agents)}, {sensor}"
                        )
                        frames = CDM.get_frames(sensor=sensor, agent=agent)[
                            1 :: args.stride
                        ]
                        for frame in tqdm(frames):
                            # get gt data
                            img_filepath = CDM.get_sensor_data_filepath(
                                frame=frame, sensor=sensor, agent=agent
                            )
                            objs = CDM.get_objects(
                                frame=frame, sensor=sensor, agent=agent
                            )
                            calib = CDM.get_calibration(
                                frame=frame, sensor=sensor, agent=agent
                            )

                            # symbolic link to image
                            img_filename = img_filepath.split("/")[-1]
                            src_img = img_filepath
                            dst_img = os.path.join(
                                img_dir,
                                split,
                                f"scene-{idx_scene}-agent-{agent}-{sensor}-{img_filename}",
                            )
                            os.symlink(src_img, dst_img)
                            images.append(
                                {
                                    "id": idx_img,
                                    "width": calib.img_shape[1],
                                    "height": calib.img_shape[0],
                                    "file_name": dst_img,
                                    "scene": scene,
                                    "frame": frame,
                                    "sensor": sensor,
                                    "agent": agent,
                                }
                            )

                            # add annotation information
                            for obj in objs:
                                # pull off boxes
                                box_3d = obj.box
                                box_2d = box_3d.project_to_2d_bbox(calib=calib)

                                # compute additional attributes
                                range_obj = obj.box.position.norm()
                                volume_3d = box_3d.volume
                                orientation_3d = box_3d.yaw
                                fraction_visible = obj.visible_fraction
                                if fraction_visible is None:
                                    raise ValueError(fraction_visible)

                                # pass targets through squeezing function with scaling functions
                                def scale(value, bounds):
                                    return (np.clip(value, *bounds) - bounds[0]) / (
                                        bounds[1] - bounds[0]
                                    )

                                # store annotation details
                                annotations.append(
                                    {
                                        "id": idx_ann,
                                        "category_id": category_ids[obj.obj_type],
                                        "iscrowd": 0,
                                        "segmentation": [[]],  # TODO: segmentation mask
                                        "area": 1000,  # TODO: segmentation area
                                        "range": range_obj,
                                        "volume_3d": volume_3d,
                                        "fraction_visible": fraction_visible,
                                        "orientation_3d": orientation_3d,
                                        "image_id": idx_img,
                                        "bbox": box_2d.box2d_xywh,
                                        "bbox_xyxy": box_2d.box2d_xyxy,
                                        "attributes": [
                                            scale(range_obj, attr_bounds["range_obj"]),
                                            scale(volume_3d, attr_bounds["volume_3d"]),
                                            scale(
                                                fraction_visible,
                                                attr_bounds["fraction_visible"],
                                            ),
                                            scale(
                                                orientation_3d,
                                                attr_bounds["orientation_3d"],
                                            ),
                                        ],
                                    }
                                )
                                idx_ann += 1

                            # convert reference frame of objects for evaluation
                            reference_objs = ReferenceFrame(
                                x=calib.reference.x,
                                q=q_cam_to_stan * calib.reference.q,
                                reference=calib.reference.reference,
                            )
                            objs_for_rel = objs.apply_and_return(
                                "change_reference",
                                reference_objs,
                                inplace=False,
                            )

                            # add relation information
                            # if idx_img == 1216:
                            #     breakpoint()
                            if args.with_relations:
                                relations[idx_img] = []
                                for idx_o1, obj1 in enumerate(objs_for_rel):
                                    for idx_o2, obj2 in enumerate(objs_for_rel):
                                        if idx_o1 != idx_o2:
                                            for REL in RELATIONS:
                                                if REL(obj1, obj2):
                                                    # add to relation list (subject, object, predicate)
                                                    # TODO: what are the subject/object indices?
                                                    relations[idx_img].append(
                                                        (
                                                            idx_o1,
                                                            idx_o2,
                                                            REL_REVINDEX[REL.name],
                                                        )
                                                    )

                            # increment and move on
                            idx_img += 1

        # package up the annotations
        annotation_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "attributes": attributes,
        }

        # package up the relations
        relation_data = {
            "relations": relations,
            "rel_categories": REL_STRINGS,
        }

        # save the annotations for this split
        ann_file = os.path.join(ann_dir, f"{split}.json")
        with open(ann_file, "w") as f:
            json.dump(annotation_data, f)
        print(f"Saved {ann_file} file")

        # save the relations for this split
        if args.with_relations:
            rel_file = os.path.join(rel_dir, f"{split}_rel.json")
            with open(rel_file, "w") as f:
                json.dump(relation_data, f)
            print(f"Saved {rel_file} file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default="/data/shared/CARLA/scenes-for-egtr/raw", type=str
    )
    parser.add_argument(
        "--output_dir", default="/data/shared/CARLA/scenes-for-egtr/processed", type=str
    )
    parser.add_argument(
        "--with_relations",
        action="store_true",
    )
    parser.add_argument("--stride", default=4, type=int)
    args = parser.parse_args()
    main(args)
