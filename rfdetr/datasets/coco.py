# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import pycocotools.mask as coco_mask

import rfdetr.datasets.transforms as T
# Import Compose directly from the module
Compose = T.Compose


def compute_multi_scale_scales(resolution, expanded_scales=False, patch_size=16, num_windows=4):
    # round to the nearest multiple of 4*patch_size to enable both patching and windowing
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    proposed_scales = [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]  # ensure minimum image size
    return proposed_scales


def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert polygon segmentation to a binary mask tensor of shape [N, H, W].
    Requires pycocotools.
    """
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            # empty segmentation for this instance
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
        except:
            rles = polygons
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, include_masks=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.include_masks = include_masks
        self.prepare = ConvertCoco(include_masks=include_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCoco(object):

    def __init__(self, include_masks=False):
        self.include_masks = include_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        
        # COCO 格式中 category_id 通常从 1 开始，但模型期望 0-based 索引
        # 需要转换为 0-based（category_id - 1）
        # 更健壮的转换：检查最小值来决定是否需要转换
        if classes.numel() > 0:
            min_class_id = classes.min().item()
            max_class_id = classes.max().item()
            # 如果最小值>=1，说明是1-based，需要转换为0-based
            if min_class_id >= 1:
                classes = classes - 1  # 转换为 0-based: 1->0, 2->1, ..., 5->4
                # 调试信息（仅在首次遇到时打印，避免日志过多）
                if not hasattr(ConvertCoco, '_class_conversion_logged'):
                    print(f"[COCO Dataset] 类别ID转换: {min_class_id}-{max_class_id} -> {min_class_id-1}-{max_class_id-1} (1-based -> 0-based)")
                    ConvertCoco._class_conversion_logged = True
            elif min_class_id == 0:
                # 类别ID从0开始，已经是0-based，无需转换
                if not hasattr(ConvertCoco, '_class_conversion_logged'):
                    print(f"[COCO Dataset] 类别ID已经是0-based ({min_class_id}-{max_class_id})，无需转换")
                    ConvertCoco._class_conversion_logged = True
            else:
                # 异常情况：负数类别ID
                print(f"[WARNING] 异常的类别ID范围: {min_class_id} ~ {max_class_id}")

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # add segmentation masks if requested, otherwise ensure consistent key when include_masks=True
        if self.include_masks:
            if len(anno) > 0 and 'segmentation' in anno[0]:
                segmentations = [obj.get("segmentation", []) for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                if masks.numel() > 0:
                    target["masks"] = masks[keep]
                else:
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            else:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target["masks"] = target["masks"].bool()

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):

    normalize = Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [resolution]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return Compose([
            T.RandomResize([resolution], max_size=1333),
            normalize,
        ])
    if image_set == 'val_speed':
        return Compose([
            T.SquareResize([resolution]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_square_div_64(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):
    """
    """

    normalize = Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    scales = [resolution]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.SquareResize(scales),
                Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.SquareResize(scales),
                ]),
            ),
            normalize,
        ])

    if image_set == 'val':
        return Compose([
            T.SquareResize([resolution]),
            normalize,
        ])
    if image_set == 'test':
        return Compose([
            T.SquareResize([resolution]),
            normalize,
        ])
    if image_set == 'val_speed':
        return Compose([
            T.SquareResize([resolution]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args, resolution):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    image_set_key = image_set.split("_")[0]
    
    # 尝试多种路径组合，按优先级顺序
    # 1. 标准COCO格式: train2017/val2017 + annotations/instances_train2017.json
    # 2. 简化COCO格式: train/val + annotations/instances_train.json (用户的数据集格式)
    
    img_folder = None
    ann_file = None
    
    # 首先尝试标准COCO格式
    if image_set_key == "train":
        img_folder = root / "train2017"
        ann_file = root / "annotations" / f'{mode}_train2017.json'
    elif image_set_key == "val":
        img_folder = root / "val2017"
        ann_file = root / "annotations" / f'{mode}_val2017.json'
    elif image_set_key == "test":
        img_folder = root / "test2017"
        ann_file = root / "annotations" / f'image_info_test-dev2017.json'
    
    # 如果标准格式不存在，尝试简化格式
    if not (img_folder.exists() and ann_file.exists()):
        if image_set_key == "train":
            img_folder = root / "train"
            ann_file = root / "annotations" / f'{mode}_train.json'
        elif image_set_key == "val":
            img_folder = root / "val"
            ann_file = root / "annotations" / f'{mode}_val.json'
        elif image_set_key == "test":
            img_folder = root / "test"
            ann_file = root / "annotations" / f'image_info_test-dev.json'
    
    # 如果还是不存在，检查标注文件是否存在（图片路径可能通过标注文件确定）
    if not ann_file.exists():
        # 尝试其他可能的标注文件路径
        if image_set_key == "train":
            ann_file = root / "annotations" / f'{mode}_train.json'
        elif image_set_key == "val":
            ann_file = root / "annotations" / f'{mode}_val.json'
        elif image_set_key == "test":
            ann_file = root / "annotations" / f'image_info_test-dev.json'
    
    if not ann_file.exists():
        raise FileNotFoundError(
            f'无法找到标注文件。尝试的路径：\n'
            f'  标准格式: {root / "annotations" / f"{mode}_{image_set_key}2017.json"}\n'
            f'  简化格式: {root / "annotations" / f"{mode}_{image_set_key}.json"}'
        )
    
    # 如果图片目录不存在，尝试从标注文件所在目录推断
    if not img_folder.exists():
        # 尝试 train/val 目录
        if image_set_key == "train":
            img_folder = root / "train"
        elif image_set_key == "val":
            img_folder = root / "val"
        elif image_set_key == "test":
            img_folder = root / "test"
        
        # 如果还是不存在，使用标注文件的父目录（对于某些特殊结构）
        if not img_folder.exists():
            img_folder = ann_file.parent
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    
    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ))
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ))
    return dataset

def build_roboflow(image_set, args, resolution):
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided Roboflow path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root /  "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }
    
    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False
    
    try:
        include_masks = args.segmentation_head
    except:
        include_masks = False

    
    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), include_masks=include_masks)
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), include_masks=include_masks)
    return dataset
