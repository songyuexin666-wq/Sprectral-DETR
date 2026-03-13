#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Spectral-DETR 矿井场景优化训练脚本
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR
# 支持 YAML 配置文件驱动的训练流程
# ------------------------------------------------------------------------

import os
import sys
import argparse
import yaml
import json
import torch
from pathlib import Path
from datetime import datetime

from rfdetr import RFDETRBase, RFDETRMedium, RFDETRLarge


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_dataset(dataset_dir: str):
    """验证数据集结构，支持多种格式"""
    print("=" * 80)
    print("验证数据集...")
    print("=" * 80)
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
    
    # 优先检查标准COCO格式 (annotations/instances_train.json)
    train_path = os.path.join(dataset_dir, "annotations", "instances_train.json")
    valid_path = os.path.join(dataset_dir, "annotations", "instances_val.json")
    
    if os.path.exists(train_path) and os.path.exists(valid_path):
        print(f"✓ 检测到标准COCO格式数据集")
        print(f"  训练集: {train_path}")
        print(f"  验证集: {valid_path}")
        return train_path, valid_path, "coco"
    
    # 检查Roboflow格式 (train/_annotations.coco.json)
    train_path = os.path.join(dataset_dir, "train", "_annotations.coco.json")
    if os.path.exists(train_path):
        valid_path = os.path.join(dataset_dir, "valid", "_annotations.coco.json")
        if not os.path.exists(valid_path):
            # 尝试 val 目录
            valid_path = os.path.join(dataset_dir, "val", "_annotations.coco.json")
            if not os.path.exists(valid_path):
                raise FileNotFoundError(f"验证集标注文件不存在")
        
        print(f"✓ 检测到Roboflow格式数据集")
        print(f"  训练集: {train_path}")
        print(f"  验证集: {valid_path}")
        return train_path, valid_path, "roboflow"
    
    raise FileNotFoundError(
        f"未找到有效的数据集格式。请确保数据集是以下格式之一：\n"
        f"  1. 标准COCO格式: {os.path.join(dataset_dir, 'annotations', 'instances_train.json')}\n"
        f"  2. Roboflow格式: {os.path.join(dataset_dir, 'train', '_annotations.coco.json')}"
    )


def get_num_classes(annotation_path: str) -> int:
    """从标注文件获取类别数"""
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    categories = data.get('categories', [])
    num_classes = len(categories)
    
    if num_classes == 0:
        raise ValueError("类别数量为0！请检查标注文件")
    
    # 检查 category_id 是否有效（允许从0开始）
    annotations = data.get('annotations', [])
    if len(annotations) > 0:
        category_ids = {ann.get('category_id') for ann in annotations}
        
        category_ids_in_cats = {cat['id'] for cat in categories}
        invalid_ids = category_ids - category_ids_in_cats
        if invalid_ids:
            raise ValueError(f"发现无效的 category_id: {invalid_ids}")
    
    print(f"✓ 检测到 {num_classes} 个类别")
    for cat in categories:
        print(f"  - {cat['id']}: {cat['name']}")
    
    return num_classes


def print_model_complexity(model, resolution: int = 576, device: str = 'cpu'):
    """打印模型参数量和 FLOPs（训练前自动调用）"""
    import torch

    # ---- 获取底层 nn.Module ----
    # model (RFDETRBase/Medium/Large) -> model.model (LwDETR wrapper) -> model.model.model (nn.Module)
    inner = None
    if hasattr(model, 'model') and hasattr(model.model, 'model'):
        inner = model.model.model
    elif hasattr(model, 'model'):
        inner = model.model
    else:
        inner = model

    # ---- 参数量 ----
    total_params = sum(p.numel() for p in inner.parameters())
    trainable_params = sum(p.numel() for p in inner.parameters() if p.requires_grad)
    params_m = total_params / 1e6

    print("\n" + "=" * 80)
    print("模型复杂度分析")
    print("=" * 80)
    print(f"  总参数量  : {params_m:.1f} M  ({total_params:,})")
    print(f"  可训练参数: {trainable_params / 1e6:.1f} M  ({trainable_params:,})")

    # ---- FLOPs（使用 torchinfo 若可用，否则用内置 benchmark.flop_count）----
    flops_g = None

    # 方案 A: torchinfo（更准确）
    try:
        from torchinfo import summary as torchinfo_summary
        _dev = torch.device(device if device != 'cuda' or torch.cuda.is_available() else 'cpu')
        dummy = torch.zeros(1, 3, resolution, resolution, device=_dev)
        # LwDETR 的 forward 接受 NestedTensor 或 tensor list；尝试 list 形式
        try:
            result = torchinfo_summary(
                inner, input_data=[[dummy[0]]],
                verbose=0, device=_dev,
            )
            flops_g = result.total_mult_adds / 1e9
        except Exception:
            result = torchinfo_summary(
                inner, input_size=(1, 3, resolution, resolution),
                verbose=0, device=_dev,
            )
            flops_g = result.total_mult_adds / 1e9
    except ImportError:
        pass

    # 方案 B: 内置 benchmark.flop_count（作为回退）
    if flops_g is None:
        try:
            from rfdetr.util.benchmark import flop_count
            _dev = 'cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
            inner_eval = inner.eval().to(_dev)
            dummy = torch.zeros(1, 3, resolution, resolution, device=_dev)
            with torch.no_grad():
                res = flop_count(inner_eval, ([dummy[0]],))
            flops_g = sum(res.values())
            inner_eval.train()
        except Exception:
            pass

    if flops_g is not None:
        print(f"  FLOPs     : {flops_g:.1f} G  (resolution={resolution}×{resolution})")
    else:
        print(f"  FLOPs     : 无法自动计算（可安装 torchinfo: pip install torchinfo）")

    print("=" * 80 + "\n")


def save_config_to_output(config: dict, output_dir: str):
    """保存配置到输出目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ 配置已保存到: {config_save_path}")


def print_experiment_info(config: dict):
    """打印实验信息（仅包含 configs/ 下的 LUE/FAFD/QCD 消融开关）"""
    model_cfg = config.get("model", {})
    name = config.get("experiment_name") or config.get("name") or "experiment"

    active = []
    if model_cfg.get("use_lue", False):
        active.append("LUE")
    if model_cfg.get("use_fafd", False):
        active.append("FAFD")
    if model_cfg.get("use_qcd", False):
        active.append("QCD")

    print("\n" + "=" * 80)
    print("实验配置")
    print("=" * 80)
    print(f"实验名称: {name}")
    print(f"激活的创新点: {active or ['无（基线）']}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RF-DETR 矿井场景优化训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径 (e.g., configs/baseline.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的checkpoint路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda/cpu，默认自动检测)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（覆盖配置文件）"
    )
    
    args = parser.parse_args()
    
    # ===== 1. 加载配置 =====
    print("=" * 80)
    print("加载配置文件...")
    print("=" * 80)
    config = load_config(args.config)
    print(f"✓ 配置文件加载成功: {args.config}")
    
    # ===== 2. 设备检测 =====
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ===== 3. 验证数据集 =====
    dataset_config = config.get('dataset', {})
    dataset_dir = dataset_config.get('dataset_dir') or dataset_config.get('coco_path') or 'datasets'
    train_path, valid_path, dataset_format = validate_dataset(dataset_dir)
    
    # 根据检测到的格式设置 dataset_file
    if dataset_format == "coco":
        dataset_file = "coco"
        # 对于COCO格式，需要设置 coco_path
        coco_path = dataset_config.get('coco_path', dataset_dir)
    else:
        dataset_file = dataset_config.get('dataset_file', 'roboflow')
        coco_path = None
    
    # ===== 4. 获取类别数 =====
    print("\n" + "=" * 80)
    print("分析数据集...")
    print("=" * 80)
    num_classes = get_num_classes(train_path)
    
    # ===== 5. 打印实验信息 =====
    print_experiment_info(config)
    
    # ===== 6. 准备输出目录 =====
    output_config = config.get('output', {})
    training_config = config.get('training', {})
    diagnostics_config = config.get('diagnostics', {})
    output_dir = (
        training_config.get('output_dir')
        or output_config.get('output_dir')
        or 'outputs/experiment'
    )
    
    # 添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    
    print(f"输出目录: {output_dir}")
    
    # 保存配置
    save_config_to_output(config, output_dir)
    
    # ===== 7. 初始化模型 =====
    print("\n" + "=" * 80)
    print("初始化模型...")
    print("=" * 80)
    
    model_config = config.get('model', {})
    
    # 根据预训练权重自动选择模型类型
    pretrain_weights = model_config.get('pretrain_weights', 'rf-detr-base.pth')
    resolution = model_config.get('resolution', 560)
    patch_size = model_config.get('patch_size', None)  # 从配置读取，如果没有则自动推断
    
    # 自动推断模型类型
    if 'medium' in pretrain_weights.lower():
        model_class = RFDETRMedium
        if patch_size is None:
            patch_size = 16  # Medium模型默认patch_size=16
        print(f"✓ 使用 RF-DETR Medium 模型 (patch_size={patch_size}, resolution={resolution})")
    elif 'large' in pretrain_weights.lower():
        model_class = RFDETRLarge
        if patch_size is None:
            patch_size = 14  # Large模型默认patch_size=14
        print(f"✓ 使用 RF-DETR Large 模型 (patch_size={patch_size}, resolution={resolution})")
    else:
        model_class = RFDETRBase
        if patch_size is None:
            patch_size = 14  # Base模型默认patch_size=14
        print(f"✓ 使用 RF-DETR Base 模型 (patch_size={patch_size}, resolution={resolution})")
    
    # 创建模型（传入configs中的创新点开关）
    model = model_class(
        num_classes=num_classes,
        resolution=resolution,
        pretrain_weights=pretrain_weights,
        patch_size=patch_size,  # 传递patch_size参数
        use_lue=model_config.get('use_lue', False),
        lue_uncertainty_weight=model_config.get('lue_uncertainty_weight', 0.5),
        lue_warmup_epochs=model_config.get('lue_warmup_epochs', 5),
        use_fafd=model_config.get('use_fafd', False),
        fafd_sparsity_weight=model_config.get('fafd_sparsity_weight', 0.0),
        use_qcd=model_config.get('use_qcd', False),
        qcd_temperature=model_config.get('qcd_temperature', 0.07),
        qcd_weight=model_config.get('qcd_weight', 0.1),
        qcd_hard_negatives_k=model_config.get('qcd_hard_negatives_k', 0),
        qcd_start_epoch=model_config.get('qcd_start_epoch', 20),
    )
    print("✓ 模型初始化完成")

    # 从模型实际配置读取 num_windows，避免与 backbone block_size 不匹配
    actual_num_windows = getattr(model.model_config, 'num_windows', 4)
    print(f"  num_windows = {actual_num_windows} (patch_size={patch_size}, block_size={patch_size * actual_num_windows})")

    # ===== 7b. 打印模型复杂度（Params & FLOPs）=====
    print_model_complexity(model, resolution=resolution, device=device)

    # ===== 8. 开始训练 =====
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    
    try:
        model.train(
            # 数据集配置
            dataset_dir=dataset_dir,
            dataset_file=dataset_file,  # 使用自动检测的格式
            coco_path=coco_path if coco_path else dataset_dir,  # COCO格式需要coco_path
            num_classes=num_classes,
            num_workers=dataset_config.get('num_workers', 4),
            square_resize_div_64=dataset_config.get('square_resize_div_64', True),
            resolution=resolution,
            patch_size=patch_size,
            num_windows=model_config.get('num_windows', actual_num_windows),
            
            # 训练配置
            epochs=training_config.get('epochs', 50),
            batch_size=training_config.get('batch_size', 16),
            grad_accum_steps=training_config.get('grad_accum_steps', 4),
            lr=training_config.get('lr', 1e-4),
            lr_encoder=training_config.get('lr_encoder', 1.5e-4),
            weight_decay=training_config.get('weight_decay', 1e-4),
            lr_drop=training_config.get('lr_drop', 40),
            warmup_epochs=training_config.get('warmup_epochs', 0.0),
            seed=args.seed if args.seed is not None else training_config.get('seed', 42),
            
            # 数据增强
            multi_scale=training_config.get('multi_scale', True),
            expanded_scales=training_config.get('expanded_scales', True),
            
            # EMA
            use_ema=training_config.get('use_ema', True),
            ema_decay=training_config.get('ema_decay', 0.993),
            ema_tau=training_config.get('ema_tau', 100),
            
            # 早停
            early_stopping=training_config.get('early_stopping', False),
            early_stopping_patience=training_config.get('early_stopping_patience', 10),
            early_stopping_min_delta=training_config.get('early_stopping_min_delta', 0.001),
            
            # 输出配置
            output_dir=output_dir,
            checkpoint_interval=training_config.get('checkpoint_interval', output_config.get('checkpoint_interval', 10)),
            tensorboard=training_config.get('tensorboard', output_config.get('tensorboard', True)),
            wandb=training_config.get('wandb', output_config.get('wandb', False)),
            run_test=training_config.get('run_test', output_config.get('run_test', True)),

            # 诊断与可视化
            diagnostics=diagnostics_config.get('enabled', True),
            diagnostics_dir=diagnostics_config.get('dir', os.path.join(output_dir, "diagnostics")),
            diagnostics_interval=diagnostics_config.get('interval', 200),
            diagnostics_max_images=diagnostics_config.get('max_images', 4),
            diagnostics_buckets=diagnostics_config.get('buckets', {
                "brightness": [0.3, 0.6],
                "contrast": [0.08, 0.16],
                "blur": [0.001, 0.004],
            }),
            
            # 设备
            device=device,
            
            # 恢复训练
            resume=args.resume,
        )
        
        print("\n" + "=" * 80)
        print("✓ 训练完成！")
        print("=" * 80)
        print(f"输出目录: {output_dir}")
        print(f"输出目录: {output_dir}")
        print(f"激活的创新点: {[k for k in ['LUE','FAFD','QCD'] if model_config.get('use_' + k.lower(), False)] or ['无（基线）']}")
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 训练过程中出现错误:")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
