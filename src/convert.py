import csv
import os
import shutil

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    fruit_train_path = "/home/alex/DATASETS/TODO/archive/AnnotatedFruitNet_FruitBox/dataset/Annotated FruitNet/images/train"
    fruit_val_path = "/home/alex/DATASETS/TODO/archive/AnnotatedFruitNet_FruitBox/dataset/Annotated FruitNet/images/val"
    fruit_box_path = "/home/alex/DATASETS/TODO/archive/AnnotatedFruitNet_FruitBox/dataset/FruitBox"
    fruit_box_w = (
        "/home/alex/DATASETS/TODO/archive/AnnotatedFruitNet_FruitBox/dataset/FruitBox/Weights.csv"
    )
    batch_size = 30
    images_ext = ".jpg"
    ann_ext = ".txt"

    ds_name_to_data = {
        "fruit net train": fruit_train_path,
        "fruit net val": fruit_val_path,
        "fruit box": fruit_box_path,
    }

    def create_ann(image_path):
        labels = []

        if ds_name == "fruit box":
            img_height = 3000
            img_wight = 4000

            weight_val = name_to_weight.get(get_file_name(image_path))
            if weight_val is not None:
                weight = sly.Tag(weight_meta, value=weight_val)
                return sly.Annotation(
                    img_size=(img_height, img_wight), labels=labels, img_tags=[weight]
                )

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        ann_path = image_path.replace("images", "labels").replace(images_ext, ann_ext)

        if file_exists(ann_path):
            with open(ann_path) as f:
                content = f.read().split("\n")

                for curr_data in content:
                    if len(curr_data) != 0:
                        curr_data = list(map(float, curr_data.split(" ")))
                        obj_class = idx_to_class[int(curr_data[0])]

                        left = int((curr_data[1] - curr_data[3] / 2) * img_wight)
                        right = int((curr_data[1] + curr_data[3] / 2) * img_wight)
                        top = int((curr_data[2] - curr_data[4] / 2) * img_height)
                        bottom = int((curr_data[2] + curr_data[4] / 2) * img_height)
                        rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                        label = sly.Label(rectangle, obj_class)
                        labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    good_apple = sly.ObjClass("good apple", sly.Rectangle)
    good_orange = sly.ObjClass("good orange", sly.Rectangle)
    bad_apple = sly.ObjClass("bad apple", sly.Rectangle)
    bad_orange = sly.ObjClass("bad orange", sly.Rectangle)
    bad_guava = sly.ObjClass("bad guava", sly.Rectangle)
    good_guava = sly.ObjClass("good guava", sly.Rectangle)
    fruit_box = sly.ObjClass("fruit box", sly.Rectangle)

    idx_to_class = {
        0: good_apple,
        1: good_guava,
        2: good_orange,
        3: bad_apple,
        4: bad_orange,
        5: bad_guava,
        6: fruit_box,
    }

    weight_meta = sly.TagMeta("weight", sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[
            good_apple,
            good_orange,
            bad_apple,
            bad_orange,
            bad_guava,
            fruit_box,
            good_guava,
        ],
        tag_metas=[weight_meta],
    )
    api.project.update_meta(project.id, meta.to_json())

    name_to_weight = {}
    with open(fruit_box_w, "r") as file:
        csvreader = csv.reader(file)
        for idx, row in enumerate(csvreader):
            if idx > 0:
                name = get_file_name(row[0])
                if name == "Image_443jpg":
                    name = "Image_443"
                name_to_weight[name] = row[1]

    for ds_name, ds_data in ds_name_to_data.items():

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_names = [
            im_name for im_name in os.listdir(ds_data) if get_file_ext(im_name) == images_ext
        ]

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(ds_data, image_name) for image_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
