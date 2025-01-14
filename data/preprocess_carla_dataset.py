import argparse
import json
import os
import shutil

from avapi.carla import CarlaScenesManager
from tqdm import tqdm


def main(args):
    """Convert Carla dataset to the COCO format"""

    # make dir structure
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    img_dir = os.path.join(args.output_dir, "images")
    ann_dir = os.path.join(args.output_dir, "annotations")
    os.makedirs(img_dir)
    os.makedirs(ann_dir)

    # set up categories
    category_ids = {
        "person": 0,
        "car": 1,
        "motorcycle": 2,
        "bicycle": 3,
        "truck": 4,
    }
    categories = [{"id": v, "name": k} for k, v in category_ids.items()]

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
                            boxes_2d = objs.apply_and_return(
                                "getattr", "box"
                            ).apply_and_return("project_to_2d_bbox", calib=calib)

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
                            for box in boxes_2d:
                                annotations.append(
                                    {
                                        "id": idx_ann,
                                        "category_id": category_ids[box.obj_type],
                                        "iscrowd": 0,
                                        "segmentation": [[]],  # TODO: segmentation mask
                                        "area": 1000,  # TODO: segmentation area
                                        "image_id": idx_img,
                                        "bbox": box.box2d_xywh,
                                    }
                                )
                                idx_ann += 1
                            idx_img += 1

        # package up the annotations
        annotation_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        # save the annotations for this split
        ann_file = os.path.join(ann_dir, f"{split}.json")
        with open(ann_file, "w") as f:
            json.dump(annotation_data, f)
        print(f"Saved {ann_file} file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default="/data/shared/CARLA/scenes-for-egtr/raw", type=str
    )
    parser.add_argument(
        "--output_dir", default="/data/shared/CARLA/scenes-for-egtr/processed", type=str
    )
    parser.add_argument("--stride", default=4, type=int)
    args = parser.parse_args()
    main(args)
