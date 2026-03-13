# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

"""
LW-DETR model and criterion classes
"""
import copy
import math
from typing import Callable
import torch
import torch.nn.functional as F
from torch import nn

from rfdetr.util import box_ops
from rfdetr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)

from rfdetr.models.backbone import build_backbone
from rfdetr.models.matcher import build_matcher
from rfdetr.models.transformer import build_transformer
from rfdetr.models.segmentation_head import SegmentationHead, get_uncertain_point_coords_with_randomness, point_sample

class LWDETR(nn.Module):
    """ This is the Group DETR v3 module that performs object detection """
    def __init__(self,
                 backbone,
                 transformer,
                 segmentation_head,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 group_detr=1,
                 two_stage=False,
                 lite_refpoint_refine=False,
                 bbox_reparam=False,
                 use_lue=False,
                 use_qcd=False,
                 use_fafd=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            group_detr: Number of groups to speed detr training. Default is 1.
            lite_refpoint_refine: TODO
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        
        # 🚀 LUE: bbox 位置回归仍保持 4 维 (cx,cy,w,h)，避免破坏 two-stage / iterative refine 逻辑。
        # 不确定性(log_var) 使用单独 head 输出 4 维。
        self.use_lue = use_lue
        self.use_qcd = use_qcd
        self.use_fafd = use_fafd
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.bbox_log_var_embed = None
        if self.use_lue:
            self.bbox_log_var_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.segmentation_head = segmentation_head
        
        query_dim=4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        # iter update
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        # 🛠️ Deep Fix: LUE 方差初始化修复 (解决"不确定性陷阱")
        # 将 log_var head 的 bias 初始化为 -5.0，对应初始方差 σ² = exp(-5.0) ≈ 0.0067
        if self.use_lue and self.bbox_log_var_embed is not None:
            nn.init.constant_(self.bbox_log_var_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_log_var_embed.layers[-1].bias.data, -5.0)

        # two_stage
        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])

        self._export = False

    def reinitialize_detection_head(self, num_classes):
        base = self.class_embed.weight.shape[0]
        num_repeats = int(math.ceil(num_classes / base))
        self.class_embed.weight.data = self.class_embed.weight.data.repeat(num_repeats, 1)
        self.class_embed.weight.data = self.class_embed.weight.data[:num_classes]
        self.class_embed.bias.data = self.class_embed.bias.data.repeat(num_repeats)
        self.class_embed.bias.data = self.class_embed.bias.data[:num_classes]
        
        if self.two_stage:
            for enc_out_class_embed in self.transformer.enc_out_class_embed:
                enc_out_class_embed.weight.data = enc_out_class_embed.weight.data.repeat(num_repeats, 1)
                enc_out_class_embed.weight.data = enc_out_class_embed.weight.data[:num_classes]
                enc_out_class_embed.bias.data = enc_out_class_embed.bias.data.repeat(num_repeats)
                enc_out_class_embed.bias.data = enc_out_class_embed.bias.data[:num_classes]

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight)

        if hs is not None:
            # bbox 位置回归 (4维)
            outputs_coord_raw = self.bbox_embed(hs)  # [num_layers, B, N, 4]

            # 🚀 LUE: 不确定性(log_var)单独 head (4维)
            outputs_log_var = None
            if self.use_lue and self.bbox_log_var_embed is not None:
                outputs_log_var = self.bbox_log_var_embed(hs)  # [num_layers, B, N, 4]
                outputs_log_var = torch.clamp(outputs_log_var, min=-7.0, max=7.0)
            
            if self.bbox_reparam:
                outputs_coord_cxcy = outputs_coord_raw[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_raw[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat(
                    [outputs_coord_cxcy, outputs_coord_wh], dim=-1
                )
            else:
                outputs_coord = (outputs_coord_raw + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)

            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(features[0].tensors, hs, samples.tensors.shape[-2:])

            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            # 🚀 LUE: 添加不确定性输出
            if self.use_lue and outputs_log_var is not None:
                out['pred_log_vars'] = outputs_log_var[-1]
            if self.use_qcd:
                out['hs'] = hs
            if self.segmentation_head is not None:
                out['pred_masks'] = outputs_masks[-1]
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(
                    outputs_class, outputs_coord, outputs_masks if self.segmentation_head is not None else None,
                    None  # LUE 只在最后一层计算：早期 decoder 层 bbox 预测不稳定，
                          # 在其上训练 uncertainty head 会引入噪声梯度，延缓收敛。
                )
            if self.use_fafd:
                fafd_loss = getattr(self.backbone[0].projector, "get_fafd_sparsity_loss", lambda: None)()
                if fafd_loss is not None:
                    out["fafd_sparsity_loss"] = fafd_loss

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)

            cls_enc = torch.cat(cls_enc, dim=1)

            if self.segmentation_head is not None:
                masks_enc = self.segmentation_head(features[0].tensors, [hs_enc,], samples.tensors.shape[-2:], skip_blocks=True)
                masks_enc = torch.cat(masks_enc, dim=1)

            if hs is not None:
                out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
                if self.segmentation_head is not None:
                    out['enc_outputs']['pred_masks'] = masks_enc
            else:
                out = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
                if self.segmentation_head is not None:
                    out['pred_masks'] = masks_enc

        return out

    def forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        # only use one group in inference
        refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
        query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight)

        outputs_masks = None

        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat(
                    [outputs_coord_cxcy, outputs_coord_wh], dim=-1
                )
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            outputs_class = self.class_embed(hs)
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(srcs[0], [hs,], tensors.shape[-2:])[0]
        else:
            assert self.two_stage, "if not using decoder, two_stage must be True"
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(srcs[0], [hs_enc,], tensors.shape[-2:], skip_blocks=True)[0]

        if outputs_masks is not None:
            return outputs_coord, outputs_class, outputs_masks
        else:
            return outputs_coord, outputs_class

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_masks, outputs_log_var=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # 🚀 LUE: 添加不确定性支持
        if outputs_masks is not None:
            if outputs_log_var is not None:
                return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c, 'pred_log_vars': d}
                        for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_masks[:-1], outputs_log_var[:-1])]
            else:
                return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_masks[:-1])]
        else:
            if outputs_log_var is not None:
                return [{'pred_logits': a, 'pred_boxes': b, 'pred_log_vars': c}
                        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_log_var[:-1])]
            else:
                return [{'pred_logits': a, 'pred_boxes': b}
                        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        """ """
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            if hasattr(self.backbone[0].encoder, 'blocks'): # Not aimv2
                if hasattr(self.backbone[0].encoder.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.blocks[i].drop_path.drop_prob = dp_rates[i]
            else: # aimv2
                if hasattr(self.backbone[0].encoder.trunk.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.trunk.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,
                num_classes,
                matcher,
                weight_dict,
                focal_alpha,
                losses,
                group_detr=1,
                sum_group_losses=False,
                use_varifocal_loss=False,
                use_position_supervised_loss=False,
                ia_bce_loss=False,
                mask_point_sample_ratio: int = 16,
                use_lue: bool = False,
                lue_uncertainty_weight: float = 0.5,
                lue_warmup_epochs: int = 5,
                use_qcd: bool = False,
                qcd_temperature: float = 0.07,
                qcd_weight: float = 0.1,
                qcd_hard_negatives_k: int = 0,
                qcd_start_epoch: int = 20,
                use_fafd: bool = False,
                # 🚀 新增：自适应参数管理
                use_adaptive_params: bool = True,
                innovation_strength: float = 1.0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            group_detr: Number of groups to speed detr training. Default is 1.
            use_adaptive_params: 是否使用自适应参数管理（强烈推荐）
            innovation_strength: 创新点整体强度 [0.5-1.5]
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss
        self.mask_point_sample_ratio = mask_point_sample_ratio

        # 🚀 CVPR创新点配置
        self.use_lue = use_lue
        self.use_qcd = use_qcd
        self.use_fafd = use_fafd
        self.current_epoch = 0

        # ⚠️ 关键修复：无条件定义所有参数（即使使用自适应参数）
        # 原因：当自适应参数返回None时（比如某个创新点未启用），需要fallback到这些默认值
        self.lue_uncertainty_weight = lue_uncertainty_weight
        self.lue_warmup_epochs = lue_warmup_epochs
        self.qcd_temperature = qcd_temperature
        self.qcd_weight = qcd_weight
        self.qcd_hard_negatives_k = qcd_hard_negatives_k
        self.qcd_start_epoch = qcd_start_epoch

        # 🚀 自适应参数管理器（三大创新点协同）
        self.use_adaptive_params = use_adaptive_params
        if use_adaptive_params:
            from rfdetr.util.adaptive_params import AdaptiveParamsManager
            self.param_manager = AdaptiveParamsManager(
                use_lue=use_lue,
                use_fafd=use_fafd,
                use_qcd=use_qcd,
                innovation_strength=innovation_strength,
                warmup_epochs=lue_warmup_epochs,
                # 从 config 传入 QCD 参数，覆盖 adaptive_params 内部默认值
                qcd_base_weight=qcd_weight,
                qcd_initial_temperature=qcd_temperature,
                qcd_hard_negatives_k=qcd_hard_negatives_k,
            )
            # 使用自适应参数
            print("✅ 使用自适应参数管理器（三大创新点协同）")
        else:
            # 使用手动配置的参数（向后兼容）
            self.param_manager = None
            print("⚠️  使用手动配置参数（不推荐，建议启用自适应参数）")

        # diagnostics buffers (main process will read these)
        self._diag_lue_err = None
        self._diag_lue_logvar = None
        self._diag_qcd_pos_sim = None
        self._diag_qcd_neg_sim = None

    def pop_diagnostics_payload(self, max_points: int = 2000):
        payload = {}

        def _cap(arr):
            if arr is None:
                return None
            if arr.numel() <= max_points:
                return arr
            idx = torch.randperm(arr.numel(), device=arr.device)[:max_points]
            return arr.flatten()[idx]

        if self._diag_lue_err is not None and self._diag_lue_logvar is not None:
            payload["lue_err"] = _cap(self._diag_lue_err).detach().cpu().numpy()
            payload["lue_logvar"] = _cap(self._diag_lue_logvar).detach().cpu().numpy()

        if self._diag_qcd_pos_sim is not None and self._diag_qcd_neg_sim is not None:
            payload["qcd_pos_sim"] = _cap(self._diag_qcd_pos_sim).detach().cpu().numpy()
            payload["qcd_neg_sim"] = _cap(self._diag_qcd_neg_sim).detach().cpu().numpy()

        self._diag_lue_err = None
        self._diag_lue_logvar = None
        self._diag_qcd_pos_sim = None
        self._diag_qcd_neg_sim = None
        return payload
    
    def set_epoch(self, epoch: int):
        """设置当前epoch，用于warm-up策略和自适应参数"""
        self.current_epoch = epoch
        if self.use_adaptive_params and self.param_manager is not None:
            self.param_manager.set_epoch(epoch)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            alpha = self.focal_alpha
            gamma = 2 
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            #init positive weights and negative weights
            pos_weights = torch.zeros_like(src_logits)
            neg_weights =  prob ** gamma

            pos_ind = tuple([id for id in idx] + [target_classes_o])

            t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights[pos_ind] = t.to(pos_weights.dtype)
            neg_weights[pos_ind] = 1 - t.to(neg_weights.dtype)
            # a reformulation of the standard loss_ce = - pos_weights * prob.log() - neg_weights * (1 - prob).log()
            # with a focus on statistical stability by using fused logsigmoid
            loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
            loss_ce = loss_ce.sum() / num_boxes

        elif self.use_position_supervised_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            # pos_ious_func = pos_ious ** 2
            pos_ious_func = pos_ious

            cls_iou_func_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind = tuple([id for id in idx] + [target_classes_o])
            cls_iou_func_targets[pos_ind] = pos_ious_func
            norm_cls_iou_func_targets = cls_iou_func_targets \
                / (cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8)
            loss_ce = position_supervised_loss(src_logits, norm_cls_iou_func_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        elif self.use_varifocal_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()

            cls_iou_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind = tuple([id for id in idx] + [target_classes_o])
            cls_iou_targets[pos_ind] = pos_ious
            loss_ce = sigmoid_varifocal_loss(src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:,:,:-1]
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes
        
        v3.1矿井优化: 自适应混合策略 (小目标用NWD, 大目标用IoU)
        v4.0 CVPR创新: LUE不确定性感知的高斯建模
        
        Args:
            outputs: 模型输出，包含 'pred_boxes' 和可选的 'pred_log_vars'
            targets: Ground Truth
            indices: 匹配结果
            num_boxes: 归一化因子
            
        Returns:
            losses: dict包含 'loss_bbox', 'loss_giou', 可选 'loss_uncertainty'
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        # Compute warmup_ratio for LUE (controls when uncertainty loss activates)
        if self.use_adaptive_params and self.param_manager is not None:
            lue_params = self.param_manager.get_lue_params()
            warmup_ratio = lue_params['warmup_ratio'] if lue_params else 0.0
        else:
            if self.lue_warmup_epochs > 0:
                warmup_ratio = min(1.0, self.current_epoch / self.lue_warmup_epochs)
            else:
                warmup_ratio = 1.0

        # GIoU loss: always standard (unchanged from baseline)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # ── L1 bbox loss ──────────────────────────────────────────────────────────
        # Base: plain L1 (identical to baseline)
        l1_per_box = F.l1_loss(src_boxes, target_boxes, reduction='none')  # (N, 4)

        if self.use_lue and 'pred_log_vars' in outputs and warmup_ratio > 0:
            src_log_vars = outputs['pred_log_vars'][idx]  # (N, 4)

            box_area = target_boxes[:, 2] * target_boxes[:, 3]  # normalized w*h

            # ① Active precision-weighted L1 for small/medium objects (core improvement)
            #
            # Motivation: LUE was purely passive (src_boxes.detach()) — it estimated
            # uncertainty but couldn't help the bbox head become more precise. This is
            # heteroscedastic regression: high-confidence predictions (low log_var →
            # high precision) receive stronger gradients → converge faster and more
            # precisely. Low-confidence predictions are softened → don't overfit to
            # noisy/occluded annotations.
            #
            # Implementation safety:
            # - log_var is DETACHED when computing precision → no circular gradient
            # - clamp(-2, 2): precision ∈ [0.14, 7.4], prevents gradient explosion
            # - precision_norm: normalized by batch mean → gradient scale ≈ plain L1
            # - only applied to small/medium objects (scale_gate) to protect APl
            # - warmup blend: pure L1 at ratio=0, precision-weighted at ratio=1
            #
            # Adaptive scale gate (dataset-agnostic):
            #   threshold = median(batch_area) * 3.0, clamped to [0.02, 0.15]
            #   小目标数据集 (Mine):   median≈0.003 → threshold≈0.009 → 几乎全部目标覆盖
            #   中等目标数据集 (ExDark): median≈0.03  → threshold≈0.09  → Car/People 覆盖
            #   大目标数据集 (COCO):   median≈0.05  → threshold≈0.15  → 自动偏大，保护 APl
            #   无需手工针对数据集调参，threshold 随 batch 分布自适应调整
            with torch.no_grad():
                median_area = box_area.median()
                dynamic_threshold = (median_area * 3.0).clamp(0.02, 0.15)
            small_mask = (box_area < dynamic_threshold).float().unsqueeze(1)  # (N, 1)
            # ① Active precision-weighted L1
            # 关键：先对 log_var 做 batch 内中心化，再 clamp，再算精度权重
            # 原因：log_var 实际均值约 -4.15，直接 clamp(-1.5, 1.1) 全部截断到同一值 →
            #       precision 全相同 → 加权退化为普通 L1。
            # 中心化后：log_var_centered ≈ N(0, 0.6²)，clamp(-1.5, 1.1) 有效区分高低置信度
            #       precision ∈ [0.33, 3.0]，高置信预测梯度最大 3x，低置信最小 0.33x
            log_var_d = src_log_vars.detach()                                    # (N, 4)
            log_var_centered = log_var_d - log_var_d.mean()                      # 中心化到 0 附近
            precision = torch.exp(-log_var_centered.clamp(min=-1.5, max=1.1))   # (N, 4), ∈[0.33, 3.0]
            precision_norm = precision / (precision.mean().detach() + 1e-6)      # (N, 4), mean≈1
            # Active weighted L1: small/medium → precision_norm * L1; large → standard L1
            weighted_l1 = small_mask * precision_norm * l1_per_box + (1 - small_mask) * l1_per_box
            final_l1 = warmup_ratio * weighted_l1 + (1 - warmup_ratio) * l1_per_box
            losses['loss_bbox'] = final_l1.sum() / num_boxes

            # ② Uncertainty 校准损失：直接回归到实际 L1 误差的 log
            # 原因：原 NLL 损失以 src_boxes.detach() 为参考，预测准确时 optimal log_var→-∞
            #       导致 variance collapse（log_var≈-4.15 全程不变，uncertainty head 失效）
            # 修复：让 log_var 直接拟合 log(实际L1误差)，误差大→高log_var，误差小→低log_var
            #       有效相关性：err-unc corr 期望从 0.50 提升到 0.65+
            with torch.no_grad():
                log_err_target = torch.log(l1_per_box.detach() + 1e-6)          # (N, 4)，目标约 -4.6
            loss_uncertainty_per_box = F.mse_loss(
                src_log_vars, log_err_target, reduction='none'
            )  # (N, 4)
            loss_uncertainty = (loss_uncertainty_per_box * small_mask).sum()
            losses['loss_uncertainty'] = loss_uncertainty * warmup_ratio / num_boxes

            losses['diag/lue_warmup_ratio'] = torch.tensor(warmup_ratio)

            with torch.no_grad():
                err = torch.abs(src_boxes - target_boxes).mean(dim=1)
                unc = src_log_vars.mean(dim=1)
                if err.numel() > 1:
                    err_c = err - err.mean()
                    unc_c = unc - unc.mean()
                    corr = (err_c * unc_c).mean() / (err_c.std() * unc_c.std() + 1e-6)
                else:
                    corr = torch.tensor(0.0, device=err.device)
                losses['diag/lue_logvar_mean'] = src_log_vars.mean()
                losses['diag/lue_logvar_std'] = src_log_vars.std()
                losses['diag/lue_err_mean'] = err.mean()
                losses['diag/lue_err_unc_corr'] = corr
                losses['diag/lue_scale_gate'] = dynamic_threshold  # 监控自适应阈值
                losses['diag/lue_small_ratio'] = small_mask.mean()  # 被 LUE 覆盖的目标比例
                self._diag_lue_err = err.detach()
                self._diag_lue_logvar = unc.detach()

        else:
            # LUE inactive or warmup not started: plain L1 (identical to baseline)
            losses['loss_bbox'] = l1_per_box.sum() / num_boxes

        return losses
    
    def loss_contrastive(self, outputs, targets, indices, num_boxes):
        """
        🚀 CVPR v4.0: QCD - Query对比去噪学习 (Supervised Contrastive Learning)
        
        改进逻辑（审稿人建议）:
        - ❌ 旧版: 所有正样本Query互相推开（破坏类内紧凑性）
        - ✅ 新版: 同类Query聚在一起，只推开背景Query
        
        目标: 在Query特征空间中
          1. 前景Query之间: 保持紧凑（同类聚在一起）
          2. 前景Query vs 背景Query: 拉大距离（去噪）
        
        理论: Supervised Contrastive Learning (InfoNCE变体)
        """
        if not self.use_qcd or 'hs' not in outputs:
            return {'loss_contrastive': torch.tensor(0.0, device=next(iter(outputs.values())).device)}

        # Delay QCD until features have stabilized (avoids noisy gradients in early training)
        if self.current_epoch < self.qcd_start_epoch:
            return {'loss_contrastive': torch.tensor(0.0, device=next(iter(outputs.values())).device)}

        # 🚀 获取自适应参数
        if self.use_adaptive_params and self.param_manager is not None:
            qcd_params = self.param_manager.get_qcd_params()
            qcd_weight = qcd_params['weight'] if qcd_params else self.qcd_weight
            qcd_temperature = qcd_params['temperature'] if qcd_params else self.qcd_temperature
            neg_pos_ratio = qcd_params['neg_pos_ratio'] if qcd_params else 10
            hard_negatives_k = qcd_params['hard_negatives_k'] if qcd_params else self.qcd_hard_negatives_k
        else:
            # 手动配置模式
            qcd_weight = self.qcd_weight
            qcd_temperature = self.qcd_temperature
            neg_pos_ratio = 10
            hard_negatives_k = self.qcd_hard_negatives_k
        
        # Use decoder representations across layers to build *always-available* positives.
        # outputs['hs'] shape: (num_layers, B, N, C)
        hs_all = outputs['hs']
        hs_last = hs_all[-1]  # (B, N, C)
        B, N, C = hs_last.shape
        
        # 获取正样本索引（匹配到目标的Query）
        idx = self._get_src_permutation_idx(indices)
        
        # 如果没有匹配，返回0
        if len(idx[0]) == 0:
            return {'loss_contrastive': torch.tensor(0.0, device=hs_last.device)}
        
        # Anchors: all matched queries at the last decoder layer.
        # This keeps QCD aligned with the original effective design: query-space
        # denoising around matched foreground slots, instead of re-defining the
        # module as a trajectory-instability selector.
        anchors = hs_last[idx]  # (P, C)

        # Per-anchor GT class label (aligned with idx order)
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # (P,)
        
        # 负样本：未匹配的Query（背景）
        pos_mask = torch.zeros(B, N, dtype=torch.bool, device=hs_last.device)
        pos_mask[idx] = True
        neg_queries_all = hs_last[~pos_mask].reshape(-1, C)  # (N_neg_all, C)
        neg_logits_all = outputs['pred_logits'][~pos_mask].reshape(-1, outputs['pred_logits'].shape[-1])  # (N_neg_all, K)

        # 如果负样本太少，返回0
        if neg_queries_all.shape[0] < 2:
            return {'loss_contrastive': torch.tensor(0.0, device=hs_last.device)}

        # L2归一化
        anchors_norm = F.normalize(anchors, dim=-1)  # (P, C)

        # Ambiguity-aware negative mining:
        # background queries that are both embedding-close and semantically ambiguous
        # are more harmful than merely similar easy negatives.
        with torch.no_grad():
            neg_prob_all = neg_logits_all.sigmoid()
            if neg_prob_all.shape[1] >= 2:
                top2 = torch.topk(neg_prob_all, k=2, dim=1).values
                neg_ambiguity = 1.0 - (top2[:, 0] - top2[:, 1])  # high => confused background
            else:
                neg_ambiguity = 1.0 - neg_prob_all[:, 0]
            neg_ambiguity = neg_ambiguity.clamp(0.0, 1.0)

        # Hard negative 选择: 必须先在全量负样本池里选 topk，再限制内存上限。
        # 修复前的 bug: 先随机下采样到 max_neg，再选 topk——
        # 随机步骤会丢弃大量全局最难负样本，导致 hard_negatives_k 形同虚设。
        if hard_negatives_k > 0:
            # 全量负样本归一化，计算每个负样本对所有 anchor 的最大余弦相似度（全局难度分）
            neg_all_norm = F.normalize(neg_queries_all, dim=-1)  # (N_neg_all, C)
            with torch.no_grad():
                sim_all = torch.matmul(anchors_norm, neg_all_norm.T)  # (P, N_neg_all)
                # 用 anchor-wise max：负样本对任意 anchor 最相似则视为最难
                hard_scores, _ = sim_all.max(dim=0)  # (N_neg_all,)
                ambiguity_bonus = 0.25 * neg_ambiguity
                hard_scores = hard_scores + ambiguity_bonus
                k_select = min(hard_negatives_k * 4, neg_queries_all.shape[0])
                _, hard_idx = torch.topk(hard_scores, k_select)
            neg_queries = neg_queries_all[hard_idx]          # 真正全局最难的负样本
        else:
            # 无 hard mining: 随机下采样以限制内存
            num_pos = anchors.shape[0]
            max_neg = num_pos * 30
            if neg_queries_all.shape[0] > max_neg:
                with torch.no_grad():
                    sample_scores = neg_ambiguity
                    _, sample_idx = torch.topk(sample_scores, k=max_neg)
                neg_queries = neg_queries_all[sample_idx]
            else:
                neg_queries = neg_queries_all

        neg_queries_norm = F.normalize(neg_queries, dim=-1)

        # diagnostics (cosine similarity, not divided by temperature)
        with torch.no_grad():
            self._diag_qcd_pos_sim = torch.matmul(anchors_norm, anchors_norm.T).flatten().detach()
            self._diag_qcd_neg_sim = torch.matmul(anchors_norm, neg_queries_norm.T).flatten().detach()

        # 🚀 使用自适应温度参数
        tau = float(qcd_temperature)

        # Negatives: background queries (last layer)
        sim_neg = torch.matmul(anchors_norm, neg_queries_norm.T) / tau  # (P, N_neg)

        # Positives (always available): same matched query across other decoder layers.
        # This avoids the "no same-class positives in a batch" failure mode common in detection.
        # ✅ P1修复: 只使用最后2-3层，防止早期层的低质量特征引入噪声
        num_layers = hs_all.shape[0]
        layer_pos_logits = None
        if num_layers > 1:
            pos_layers = []
            # 只使用最后3层（如果有的话）
            start_layer = max(0, num_layers - 3)
            for l in range(start_layer, num_layers - 1):
                pos_layers.append(hs_all[l][idx])  # (P, C)
            pos_layers = torch.stack(pos_layers, dim=1)  # (P, L-1, C)
            pos_layers = F.normalize(pos_layers, dim=-1)
            # sim: (P, L-1)
            layer_pos_logits = torch.einsum("pc,plc->pl", anchors_norm, pos_layers) / tau

        # Optional extra positives: other matched queries of the same class (last layer)
        same_class_pos_logits = None
        P = anchors_norm.shape[0]
        # per_anchor_has_same_class: True if this anchor has ≥1 other matched query of the same class
        # in the current batch. False for rare-class anchors (class_12/13 have <2% chance per batch).
        per_anchor_has_same_class = torch.zeros(P, dtype=torch.bool, device=hs_last.device)
        if P > 1:
            sim_pos_all = torch.matmul(anchors_norm, anchors_norm.T) / tau  # (P, P)
            same_class = target_classes.view(-1, 1).eq(target_classes.view(1, -1))
            eye = torch.eye(P, dtype=torch.bool, device=hs_last.device)
            pos_mask_mat = same_class & (~eye)
            per_anchor_has_same_class = pos_mask_mat.any(dim=1)  # (P,) — False for isolated rare-class anchors
            if pos_mask_mat.any():
                same_class_pos_logits = sim_pos_all.masked_fill(~pos_mask_mat, float("-inf"))

        # If we have no positives at all (rare: num_layers==1 and no same-class), return 0
        if layer_pos_logits is None and same_class_pos_logits is None:
            loss_contrastive = torch.tensor(0.0, device=hs_last.device)
        else:
            # numerator: logsumexp over positives (layer positives + same-class positives)
            nums = []
            if layer_pos_logits is not None:
                nums.append(torch.logsumexp(layer_pos_logits, dim=1))  # (P,)
            if same_class_pos_logits is not None:
                nums.append(torch.logsumexp(same_class_pos_logits, dim=1))  # (P,)
            log_num = torch.logaddexp(nums[0], nums[1]) if len(nums) == 2 else nums[0]

            # negatives: optionally hard-neg topk per anchor
            if hard_negatives_k > 0 and sim_neg.shape[1] > hard_negatives_k:
                neg_topk, _ = torch.topk(sim_neg, hard_negatives_k, dim=1)  # (P, K)
                log_neg = torch.logsumexp(neg_topk, dim=1)  # (P,)
            else:
                log_neg = torch.logsumexp(sim_neg, dim=1)  # (P,)

            log_den = torch.logaddexp(log_num, log_neg)
            loss_per_anchor = -(log_num - log_den)  # (P,), per-anchor InfoNCE

            # ── 稀有类保护 ──────────────────────────────────────────────────
            # 问题根源: class_12 只有 10 个实例(2.4%的图), class_13 只有 6 个(1.5%),
            # 且同一张图从不出现 2 个实例。因此 99%+ 的 batch 里这些类没有
            # same-class 正样本对，但 QCD 仍然对它们施加 200 个 hard negative。
            # 这造成"0正样本 vs 200负样本"的极端不平衡，导致稀有类 query 被
            # 推挤到 embedding 空间的角落，表现为晚期训练不稳定/过拟合。
            # Baseline 没有 QCD，因此不受此影响。
            #
            # 修复: per-anchor 加权——没有 same-class 正样本的 anchor（稀有类）
            # 使用 rare_cls_weight 倍的 loss，减少 hard negative 的不平衡压迫。
            # 仍保留部分对比信号（≠0），完全跳过会让稀有类完全脱离对比学习。
            # ─────────────────────────────────────────────────────────────────
            rare_cls_weight = 0.2  # 稀有类（无 same-class pos）的 loss 缩放系数
            anchor_weights = torch.where(
                per_anchor_has_same_class,
                torch.ones(P, device=hs_last.device),
                torch.full((P,), rare_cls_weight, device=hs_last.device),
            )
            # 加权平均（保证梯度量纲正确）
            loss_contrastive = (loss_per_anchor * anchor_weights).sum() / anchor_weights.sum()

        # Diagnostics payload
        diag = {
            'diag/qcd_pos': torch.tensor(float(P), device=hs_last.device, dtype=torch.float32),
            'diag/qcd_neg': torch.tensor(float(neg_queries.shape[0]), device=hs_last.device, dtype=torch.float32),
            'diag/qcd_hard_neg': torch.tensor(float(hard_negatives_k), device=hs_last.device, dtype=torch.float32),
            'diag/qcd_layers': torch.tensor(float(num_layers), device=hs_last.device, dtype=torch.float32),
            'diag/qcd_loss_raw': loss_contrastive.detach(),
            'diag/qcd_weight': torch.tensor(float(qcd_weight), device=hs_last.device, dtype=torch.float32),
            'diag/qcd_temperature': torch.tensor(float(qcd_temperature), device=hs_last.device, dtype=torch.float32),
        }
        # 🚀 使用自适应权重
        out = {'loss_contrastive': loss_contrastive * qcd_weight}
        out.update(diag)
        return out
    
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute BCE-with-logits and Dice losses for segmentation masks on matched pairs.
        Expects outputs to contain 'pred_masks' of shape [B, Q, H, W] and targets with key 'masks'.
        """
        assert 'pred_masks' in outputs, "pred_masks missing in model outputs"
        pred_masks = outputs['pred_masks']  # [B, Q, H, W]
        # gather matched prediction masks
        idx = self._get_src_permutation_idx(indices)
        src_masks = pred_masks[idx]  # [N, H, W]
        # handle no matches
        if src_masks.numel() == 0:
            return {
                'loss_mask_ce': src_masks.sum(),
                'loss_mask_dice': src_masks.sum(),
            }
        # gather matched target masks
        target_masks = torch.cat([t['masks'][j] for t, (_, j) in zip(targets, indices)], dim=0)  # [N, Ht, Wt]
        
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()

        num_points = max(src_masks.shape[-2], src_masks.shape[-2] * src_masks.shape[-1] // self.mask_point_sample_ratio)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                num_points,
                3,
                0.75,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
                mode="nearest",
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_ce": sigmoid_ce_loss_jit(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss_jit(point_logits, point_labels, num_boxes),
        }

        del src_masks
        del target_masks
        return losses
    
 
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'contrastive': self.loss_contrastive,  # 🚀 CVPR v4.0: QCD
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'fafd_sparsity_loss' in outputs:
            losses['loss_fafd_sparsity'] = outputs['fafd_sparsity_loss']
        
        # 🚀 CVPR v4.0: QCD - 如果启用，添加对比学习Loss
        if self.use_qcd:
            losses.update(self.get_loss('contrastive', outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def sigmoid_varifocal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    focal_weight = targets * (targets > 0.0).float() + \
            (1 - alpha) * (prob - targets).abs().pow(gamma) * \
            (targets <= 0.0).float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * focal_weight

    return loss.mean(1).sum() / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (torch.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
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
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
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
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_masks = outputs.get('pred_masks', None)
        out_log_vars = outputs.get('pred_log_vars', None)  # 🚀 LUE: 获取不确定性（log σ²）

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        
        # 🛠️ Step 2: 修改推理逻辑 - 解决"过度惩罚"问题
        # 
        # 【专家分析】高不确定性不代表不是物体，只代表位置可能不准
        # 直接惩罚Score会导致"诚实"的高不确定性预测被错误过滤
        #
        # 【方案A - 保守策略】推理时完全忽略不确定性，只看位置均值
        # 如果这比Baseline好，说明训练策略是对的，只是后处理逻辑错了
        #
        # 【方案B - 进阶策略】未来可以在NMS中使用不确定性，或用于Soft-NMS
        # 当前采用方案A，确保不误杀检测结果
        #
        # 注意：如果需要使用不确定性，建议：
        # 1. 用于Soft-NMS的权重调整（高不确定性框更容易被抑制）
        # 2. 用于可视化（显示不确定性热图）
        # 3. 用于后处理（在NMS后根据不确定性微调框的位置，但不要降低Score）
        #
        # 暂时注释掉不确定性惩罚，让模型专注于位置回归的准确性
        # if out_log_vars is not None:
        #     uncertainty = out_log_vars.mean(dim=-1)
        #     uncertainty_clamped = torch.clamp(uncertainty, min=-7.0, max=7.0)
        #     penalty = torch.exp(-0.5 * torch.exp(uncertainty_clamped).clamp(max=1.0))
        #     prob = prob * penalty.unsqueeze(-1)
        
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # LUE: 为被选中的 top-K queries 计算每个检测框的不确定性（标量）
        # 使用 4 个坐标 log_var 的均值作为每个框的整体不确定性度量
        uncertainties = None
        if out_log_vars is not None:
            # out_log_vars: [B, Q, 4]，与 out_bbox 对齐
            gathered_logvars = torch.gather(
                out_log_vars, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_log_vars.shape[-1])
            )  # [B, K, 4]
            # 对 4 个坐标的 log_var 取均值，得到每个框一个标量
            uncertainties = gathered_logvars.mean(dim=-1)  # [B, K]

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # Optionally gather masks corresponding to the same top-K queries and resize to original size
        results = []
        if out_masks is not None:
            for i in range(out_masks.shape[0]):
                res_i = {'scores': scores[i], 'labels': labels[i], 'boxes': boxes[i]}
                if uncertainties is not None:
                    res_i['uncertainty'] = uncertainties[i]
                k_idx = topk_boxes[i]
                masks_i = torch.gather(
                    out_masks[i],
                    0,
                    k_idx.unsqueeze(-1).unsqueeze(-1).repeat(
                        1, out_masks.shape[-2], out_masks.shape[-1]
                    ),
                )  # [K, Hm, Wm]
                h, w = target_sizes[i].tolist()
                masks_i = F.interpolate(
                    masks_i.unsqueeze(1),
                    size=(int(h), int(w)),
                    mode='bilinear',
                    align_corners=False,
                )  # [K,1,H,W]
                res_i['masks'] = masks_i > 0.0
                results.append(res_i)
        else:
            for s, l, b, u in zip(
                scores,
                labels,
                boxes,
                uncertainties if uncertainties is not None else [None] * scores.shape[0],
            ):
                res = {'scores': s, 'labels': l, 'boxes': b}
                if u is not None:
                    res['uncertainty'] = u
                results.append(res)

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.num_classes + 1
    device = torch.device(args.device)


    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=args.shape if hasattr(args, 'shape') else (args.resolution, args.resolution) if hasattr(args, 'resolution') else (640, 640),
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.pretrain_weights is None,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
        use_fafd=getattr(args, 'use_fafd', False),
        fafd_sparsity_weight=getattr(args, 'fafd_sparsity_weight', 0.0),
        # 保持可配置但不强迫暴露一堆“黑盒参数”：
        # - 默认使用 MultiScaleProjector 的保守缺省值（alpha=0.15, target_keep=0.8, entropy_weight=0）
        # - 如果你确实需要调过滤强度，只改 fafd_alpha 即可
        fafd_alpha=getattr(args, 'fafd_alpha', 0.15),
    )
    if args.encoder_only:
        return backbone[0].encoder, None, None
    if args.backbone_only:
        return backbone, None, None

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)

    segmentation_head = SegmentationHead(args.hidden_dim, args.dec_layers, downsample_ratio=args.mask_downsample_ratio) if args.segmentation_head else None

    # 🚀 CVPR创新: LUE参数
    use_lue = getattr(args, 'use_lue', False)
    use_qcd = getattr(args, 'use_qcd', False)
    use_fafd = getattr(args, 'use_fafd', False)
    
    model = LWDETR(
        backbone,
        transformer,
        segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
        use_lue=use_lue,
        use_qcd=use_qcd,
        use_fafd=use_fafd,
    )
    return model

def build_criterion_and_postprocessors(args):
    device = torch.device(args.device)
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.segmentation_head:
        weight_dict['loss_mask_ce'] = args.mask_ce_loss_coef
        weight_dict['loss_mask_dice'] = args.mask_dice_loss_coef

    # Read innovation flags before building aux_weight_dict so aux layers inherit them
    use_lue = getattr(args, 'use_lue', False)
    lue_uncertainty_weight = getattr(args, 'lue_uncertainty_weight', 0.5)
    lue_warmup_epochs = getattr(args, 'lue_warmup_epochs', 15)
    use_qcd = getattr(args, 'use_qcd', False)
    qcd_temperature = getattr(args, 'qcd_temperature', 0.15)
    qcd_weight = getattr(args, 'qcd_weight', 0.3)
    qcd_hard_negatives_k = getattr(args, 'qcd_hard_negatives_k', 128)
    qcd_start_epoch = getattr(args, 'qcd_start_epoch', 20)
    fafd_sparsity_weight = getattr(args, 'fafd_sparsity_weight', 0.0)

    # LUE: register uncertainty as independent auxiliary loss
    if use_lue:
        lue_uncertainty_coef = getattr(args, 'lue_uncertainty_coef', 0.5)
        weight_dict['loss_uncertainty'] = lue_uncertainty_coef

    if fafd_sparsity_weight and fafd_sparsity_weight > 0:
        weight_dict['loss_fafd_sparsity'] = fafd_sparsity_weight

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.segmentation_head:
        losses.append('masks')

    try:
        sum_group_losses = args.sum_group_losses
    except:
        sum_group_losses = False

    if args.segmentation_head:
        criterion = SetCriterion(args.num_classes + 1, matcher=matcher, weight_dict=weight_dict,
                                focal_alpha=args.focal_alpha, losses=losses, 
                                group_detr=args.group_detr, sum_group_losses=sum_group_losses,
                                use_varifocal_loss = args.use_varifocal_loss,
                                use_position_supervised_loss=args.use_position_supervised_loss,
                                ia_bce_loss=args.ia_bce_loss,
                                mask_point_sample_ratio=args.mask_point_sample_ratio,
                                use_lue=use_lue,
                                lue_uncertainty_weight=lue_uncertainty_weight,
                                lue_warmup_epochs=lue_warmup_epochs,
                                use_qcd=use_qcd,
                                qcd_temperature=qcd_temperature,
                                qcd_weight=qcd_weight,
                                qcd_hard_negatives_k=qcd_hard_negatives_k,
                                qcd_start_epoch=qcd_start_epoch)
    else:
        criterion = SetCriterion(args.num_classes + 1, matcher=matcher, weight_dict=weight_dict,
                                focal_alpha=args.focal_alpha, losses=losses,
                                group_detr=args.group_detr, sum_group_losses=sum_group_losses,
                                use_varifocal_loss = args.use_varifocal_loss,
                                use_position_supervised_loss=args.use_position_supervised_loss,
                                ia_bce_loss=args.ia_bce_loss,
                                use_lue=use_lue,
                                lue_uncertainty_weight=lue_uncertainty_weight,
                                lue_warmup_epochs=lue_warmup_epochs,
                                use_qcd=use_qcd,
                                qcd_temperature=qcd_temperature,
                                qcd_weight=qcd_weight,
                                qcd_hard_negatives_k=qcd_hard_negatives_k,
                                qcd_start_epoch=qcd_start_epoch)
    criterion.to(device)
    postprocess = PostProcess(num_select=args.num_select)

    return criterion, postprocess
