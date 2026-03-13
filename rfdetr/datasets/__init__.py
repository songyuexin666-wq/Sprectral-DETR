# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

import torch.utils.data
import torchvision

from .coco import build as build_coco
from .o365 import build_o365
from .coco import build_roboflow


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args, resolution):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args, resolution)
    if args.dataset_file == 'o365':
        return build_o365(image_set, args, resolution)
    if args.dataset_file == 'roboflow':
        return build_roboflow(image_set, args, resolution)
    raise ValueError(f'dataset {args.dataset_file} not supported')
