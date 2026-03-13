# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

import inspect
from typing import Dict, List

import torch
from torch import nn

from rfdetr.util.misc import NestedTensor
from rfdetr.models.position_encoding import build_position_encoding
from rfdetr.models.backbone.backbone import *
from typing import Callable

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor):
        """ """
        x = self[0](tensor_list)
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if (
                hasattr(m, "export")
                and isinstance(m.export, Callable)
                and hasattr(m, "_export")
                and not m._export
            ):
                m.export()

    def forward_export(self, inputs: torch.Tensor):
        feats, masks = self[0](inputs)
        poss = []
        for feat, mask in zip(feats, masks):
            poss.append(self[1](mask, align_dim_orders=False).to(feat.dtype))
        return feats, None, poss


def build_backbone(
    encoder,
    vit_encoder_num_layers,
    pretrained_encoder,
    window_block_indexes,
    drop_path,
    out_channels,
    out_feature_indexes,
    projector_scale,
    use_cls_token,
    hidden_dim,
    position_embedding,
    freeze_encoder,
    layer_norm,
    target_shape,
    rms_norm,
    backbone_lora,
    force_no_pretrain,
    gradient_checkpointing,
    load_dinov2_weights,
    patch_size,
    num_windows,
    positional_encoding_size,
    use_fafd=False,  # 🚀 CVPR v4.0: 是否使用频域门控特征解耦
    fafd_sparsity_weight=0.0,
    fafd_alpha: float = 0.5,
    **_unused_kwargs,
):
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(hidden_dim, position_embedding)

    backbone_params = inspect.signature(Backbone.__init__).parameters
    backbone_kwargs = dict(
        encoder=encoder,
        pretrained_encoder=pretrained_encoder,
        window_block_indexes=window_block_indexes,
        drop_path=drop_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        use_cls_token=use_cls_token,
        layer_norm=layer_norm,
        freeze_encoder=freeze_encoder,
        target_shape=target_shape,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        gradient_checkpointing=gradient_checkpointing,
        load_dinov2_weights=load_dinov2_weights,
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
    )
    if "fafd_alpha" in backbone_params:
        backbone_kwargs.update(
            use_fafd=use_fafd,
            fafd_sparsity_weight=fafd_sparsity_weight,
            fafd_alpha=fafd_alpha,
        )

    backbone = Backbone(**backbone_kwargs)

    model = Joiner(backbone, position_embedding)
    return model
