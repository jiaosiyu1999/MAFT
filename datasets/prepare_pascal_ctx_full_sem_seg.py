"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/NVlabs/ODISE/blob/main/datasets/prepare_pascal_ctx_full_sem_seg.py
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import scipy.io as sio

import tqdm


def generate_labels(mat_file, out_dir):

    mat = sio.loadmat(mat_file)
    label_map = mat["LabelMap"]
    assert label_map.dtype == np.uint16
    label_map[label_map == 0] = 65535
    label_map = label_map - 1
    label_map[label_map == 65534] = 65535

    out_file = out_dir / Path(mat_file.name).with_suffix(".tif")
    Image.fromarray(label_map).save(out_file)


if __name__ == "__main__":
    voc_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "VOCdevkit/VOC2010"
    mat_dir = voc_dir / "trainval"
    
    file_names = list((voc_dir / "val" / "image").glob("*.jpg"))
    output_ann_dir = voc_dir / "annotations_detectron2_ovs" / 'pc459_val'

    output_ann_dir.mkdir(parents=True, exist_ok=True)

    for file_name in tqdm.tqdm(file_names):
        mat_file_path = mat_dir / f"{file_name.stem}.mat"

        generate_labels(mat_file_path, output_ann_dir)