import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# cls_name 
CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tv",
)

import random
color = [[random.randint(0,255), random.randint(0,255), random.randint(0,255)] for i in range(len(CLASS_NAMES))]

def _get_voc_meta(cat_list):
    # colorlist = [i['color'] for i in color]
    colorlist = color
    ret = {
        "stuff_classes": cat_list,
        "stuff_colors": colorlist,
    }
    return ret


def register_all_voc_11k(root):
    # root = os.path.join(root, "VOC2012")
    meta = _get_voc_meta(CLASS_NAMES)

    for name, image_dirname, sem_seg_dirname in [
        ("val", "JPEGImages", "annotations_ovs/val"),
    ]:
        image_dir = os.path.join(root, 'img')
        gt_dir = os.path.join(root, 'gt')
        all_name = f"my_test_dataset"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets/my_test_dataset")
register_all_voc_11k(_root)
