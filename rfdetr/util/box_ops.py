# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x, indexing='ij')

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


# ==================================================================================
# 🚀 CVPR Level Innovation: LUE (Localization Uncertainty Estimation)
# ==================================================================================

def gaussian_focal_loss(pred_boxes, pred_log_vars, target_boxes, reduction='mean'):
    """
    高斯Focal Loss - 基于不确定性的自适应定位Loss
    
    创新点:
    - 将bbox从确定值升级为概率分布 N(μ, σ²)
    - 模型自动学习"哪里看得清"（低σ）vs"哪里看不清"（高σ）
    - 不确定性高的地方，Loss自动放宽要求（容错）
    
    理论基础:
    - 负对数似然 (Negative Log-Likelihood):
      -log p(y|x) = 1/(2σ²) * ||y - μ||² + 1/2 * log(σ²) + const
    - 等价于加权L1 Loss + 正则项
    
    Args:
        pred_boxes: (N, 4) 预测的bbox [cx, cy, w, h]
        pred_log_vars: (N, 4) 预测的log(σ²)，即不确定性
        target_boxes: (N, 4) Ground Truth bbox
        reduction: 'mean' or 'sum' or 'none'
    
    Returns:
        loss: scalar or (N, 4) tensor
        
    物理意义:
        当 σ² → 0 (确定): Loss ≈ L1 Loss (要求精确)
        当 σ² → ∞ (不确定): Loss ≈ log(σ²) (允许偏差，但惩罚过度不确定)
    """
    # 1. 计算precision (精度 = 1/方差)
    # 🛡️ 数值稳定性保护: clamp log_var到合理范围，防止exp溢出
    # log_var范围: [-7, 7] 对应 σ²范围: [exp(-7)≈0.001, exp(7)≈1096]
    pred_log_vars = torch.clamp(pred_log_vars, min=-7.0, max=7.0)
    # 使用exp(-log_var) = 1/σ² 确保数值稳定
    precision = torch.exp(-pred_log_vars)  # (N, 4)
    
    # 2. 计算预测误差
    error = torch.abs(pred_boxes - target_boxes)  # (N, 4) L1距离
    
    # 3. 高斯Focal Loss公式
    # loss = precision * error + 0.5 * log_var
    # 第一项: 精度加权误差（不确定性高→权重低→容错）
    # 第二项: 正则项（防止模型逃避，将所有位置都标记为"不确定"）
    loss = precision * error + 0.5 * pred_log_vars  # (N, 4)
    
    # 4. Reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def gaussian_focal_loss_coordinate_decoupled(pred_boxes, pred_log_vars, target_boxes, 
                                               reduction='mean',
                                               center_weight=1.0,
                                               size_weight=0.8):
    """
    🛠️ CVPR专家级改进: 坐标解耦的高斯Focal Loss
    
    核心思想:
    - 小目标的中心点(cx, cy)通常比宽高(w, h)预测得更准
    - 对中心点和宽高分别给予不同约束强度
    - 中心点：强约束（低容忍度）
    - 宽高：弱约束（高容忍度，因为小目标的宽高容易受标注误差影响）
    
    理论基础:
    - 对于矿井小目标，1-2个像素的偏差在宽高上影响更大
    - 中心点定位是关键，方差容易收敛
    - 宽高预测难，允许更大的不确定性
    
    Args:
        pred_boxes: (N, 4) 预测的bbox [cx, cy, w, h]
        pred_log_vars: (N, 4) 预测的log(σ²)
        target_boxes: (N, 4) Ground Truth bbox
        reduction: 'mean' or 'sum' or 'none'
        center_weight: 中心点坐标的权重 (默认1.0，强约束)
        size_weight: 宽高坐标的权重 (默认0.8，弱约束)
    
    Returns:
        loss: scalar or (N, 4) tensor
    """
    # 1. 数值稳定性保护
    pred_log_vars = torch.clamp(pred_log_vars, min=-7.0, max=7.0)
    precision = torch.exp(-pred_log_vars)  # (N, 4)
    
    # 2. 计算预测误差
    error = torch.abs(pred_boxes - target_boxes)  # (N, 4)
    
    # 3. 🛠️ 坐标解耦：分别处理中心点和宽高
    # 前2维: 中心点 (cx, cy) - 强约束
    # 后2维: 宽高 (w, h) - 弱约束
    center_error = error[:, :2]  # (N, 2)
    center_precision = precision[:, :2]  # (N, 2)
    center_log_vars = pred_log_vars[:, :2]  # (N, 2)
    
    size_error = error[:, 2:]  # (N, 2)
    size_precision = precision[:, 2:]  # (N, 2)
    size_log_vars = pred_log_vars[:, 2:]  # (N, 2)
    
    # 4. 分别计算中心点和宽高的loss，应用不同权重
    loss_center = center_precision * center_error + 0.5 * center_log_vars  # (N, 2)
    loss_size = size_precision * size_error + 0.5 * size_log_vars  # (N, 2)
    
    # 5. 加权组合
    loss = torch.cat([
        loss_center * center_weight,
        loss_size * size_weight
    ], dim=-1)  # (N, 4)
    
    # 6. Reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def uncertainty_aware_iou_loss(pred_boxes, pred_log_vars, target_boxes, 
                                 use_giou=True, uncertainty_weight=0.5):
    """
    结合不确定性的IoU Loss
    
    创新点:
    - 将高斯Focal Loss与GIoU结合
    - 双重优化: 位置精度 + 几何重叠
    
    Args:
        pred_boxes: (N, 4) [cx, cy, w, h] 归一化
        pred_log_vars: (N, 4) log(σ²)
        target_boxes: (N, 4)
        use_giou: 是否使用GIoU (vs 普通IoU)
        uncertainty_weight: 不确定性Loss的权重
        
    Returns:
        loss_dict: {'loss_giou': scalar, 'loss_uncertainty': scalar}
    """
    # 1. 几何Loss (GIoU)
    if use_giou:
        iou_matrix = generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        )
        loss_giou = 1.0 - torch.diag(iou_matrix).mean()
    else:
        iou_matrix = box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        )[0]
        loss_giou = 1.0 - torch.diag(iou_matrix).mean()
    
    # 2. 不确定性Loss (高斯Focal)
    loss_uncertainty = gaussian_focal_loss(pred_boxes, pred_log_vars, target_boxes)
    
    # 3. 组合（两个Loss都重要）
    total_loss = loss_giou + uncertainty_weight * loss_uncertainty
    
    return {
        'loss_giou': loss_giou,
        'loss_uncertainty': loss_uncertainty,
        'loss_total': total_loss
    }
