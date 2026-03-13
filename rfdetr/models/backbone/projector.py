# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

"""
Projector
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """
        LayerNorm forward
        TODO: this is a hack to avoid overflow when using fp16
        """
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.size(3),), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def get_activation(name, inplace=False):
    """ get activation """
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name in ["LeakyReLU", 'leakyrelu', 'lrelu']:
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# ============================================================================
# 🚀 CVPR v4.0: FAFD (Frequency-Aware Feature Disentanglement)
# ============================================================================

class FrequencyGatedBlock(nn.Module):
    """
    频域门控特征解耦模块 (CVPR v4.0)
    
    核心创新:
        - FFT频域学习 ✅ 数据驱动
        
    理论基础:
        - 傅里叶变换: f(x,y) → F(u,v) (空域→频域)
        - 信号分析: 
            * 物体边缘: 低频+中频(有规律的周期性)
            * 粉尘噪声: 全频段(随机，无规律)
        - 可学习滤波: 网络自动发现最优频率保留策略
        
    创新点:
        1. 空域→频域: 使用FFT转换到频域进行分析
        2. 可学习Mask: 网络学习M(u,v)决定保留/抑制哪些频率
        3. 自适应滤波: 清晰场景保留高频，粉尘场景抑制
        4. 物理可解释: 基于信号处理理论，非黑盒
        
    CVPR价值:
        - 数据驱动的频域滤波
        - 信号处理+深度学习的融合
        - 泛化到所有噪声场景（医疗、水下、夜视）
    """
    def __init__(
        self,
        channels,
        reduction=4,
        alpha: float = 0.5,
        target_keep: float = 0.5,
        entropy_weight: float = 0.0,
    ):
        super().__init__()

        # 频域门控主干网络（不含 Sigmoid，FiLM 调制后再 Sigmoid）
        # 初始化: 最后一个 Conv bias=4.0，使初始 gate ≈ Sigmoid(4.0) ≈ 0.982（近恒等）
        self.freq_gate_body = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            # 无 Sigmoid: FiLM 调制后再统一过 Sigmoid
        )
        nn.init.constant_(self.freq_gate_body[2].bias, 4.0)
        nn.init.zeros_(self.freq_gate_body[2].weight)

        # 场景自适应分支 (FiLM: Feature-wise Linear Modulation)
        # 输入: 全局幅度谱均值 → 预测 (gamma, beta) 调制 gate 响应
        # 物理意义: 降质场景（粉尘/低光，高噪声能量）→ 更激进的滤波
        #           清晰场景 → 保守滤波，保留高频边缘细节
        # 初始化: gamma=1, beta=0，等价于恒等变换，不影响训练初期稳定性
        self.scene_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, 1, 1) → 全局场景统计
            nn.Flatten(),              # (B, C)
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, 2 * channels),  # → [gamma (C), beta (C)]
        )
        # gamma 初始化为 1（不缩放），beta 初始化为 0（不偏移）
        nn.init.zeros_(self.scene_encoder[4].weight)
        nn.init.constant_(self.scene_encoder[4].bias[:channels], 1.0)   # gamma = 1
        nn.init.constant_(self.scene_encoder[4].bias[channels:], 0.0)   # beta  = 0

        self.channels = channels
        self.alpha = float(alpha)
        self.target_keep = float(target_keep)
        self.entropy_weight = float(entropy_weight)
        self._last_gate = None
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征图

        Returns:
            out: (B, C, H, W) 频域滤波后的特征图
        """
        B, C, H, W = x.shape

        # 梯度累积时避免复用上一轮的 _last_gate，防止 "backward through the graph a second time"
        self._last_gate = None

        # 🛠️ 性能修复: 小特征图跳过FFT，避免量化噪声
        # 对于16×16以下的小特征图，FFT会引入噪声，破坏小目标特征
        if H * W < 32 * 32:
            return x  # 直接返回原特征，不做频域滤波

        # 1) FFT: 空域 → 频域（在 fp32 中更稳）
        # x_fft: (B, C, H, W//2+1) complex
        x_fft = torch.fft.rfft2(x.float(), norm='backward')

        # 2) 频域幅度谱
        amp = torch.log1p(torch.abs(x_fft))  # real, (B, C, H, W//2+1)
        amp_dtype = amp.to(dtype=x.dtype)

        # 3) 场景自适应 FiLM: 用全局幅度统计生成 (gamma, beta) 调制 gate
        # 降质场景（高噪声能量）和清晰场景会产生不同的 gamma/beta，
        # 使 gate 能自适应地决定过滤强度，而不是学一个对所有场景的平均响应。
        scene = self.scene_encoder(amp_dtype)         # (B, 2C)
        gamma = scene[:, :self.channels].view(B, self.channels, 1, 1)   # (B, C, 1, 1)
        beta  = scene[:, self.channels:].view(B, self.channels, 1, 1)   # (B, C, 1, 1)

        # 4) 频域门控: gate_body 输出原始 logit，FiLM 调制后过 Sigmoid
        raw_gate = self.freq_gate_body(amp_dtype)             # (B, C, H, W//2+1)
        gate_freq = torch.sigmoid(raw_gate * gamma + beta)    # (B, C, H, W//2+1), in [0,1]
        self._last_gate = gate_freq

        # 5) 频域逐点门控（broadcast 到 complex）
        x_filtered = x_fft * gate_freq.float()

        # 6) iFFT: 频域 → 空域
        x_out = torch.fft.irfft2(x_filtered, s=(H, W), norm='backward')

        # 7) Blend（关键：避免"叠加放大/不净化"）
        # out = (1-alpha)*x + alpha*x_out
        alpha = self.alpha
        x_out = x_out.to(dtype=x.dtype)
        return x * (1.0 - alpha) + x_out * alpha

    def get_gate_stats(self):
        if self._last_gate is None:
            return None
        gate = self._last_gate.detach()
        mean = gate.mean().item()
        std = gate.std().item()
        sparsity = (gate < 0.1).float().mean().item()
        return {"mean": mean, "std": std, "sparsity": sparsity}

    def get_sparsity_loss(self):
        if self._last_gate is None:
            return None
        # Regularize gate to be selective but not collapsed.
        # - keep ratio: encourage mean(gate) ≈ target_keep
        # - entropy: encourage gate to be closer to {0,1} (low entropy)
        gate = self._last_gate
        keep_loss = (gate.mean() - self.target_keep) ** 2

        if self.entropy_weight > 0:
            eps = 1e-6
            g = gate.clamp(min=eps, max=1.0 - eps)
            entropy = -(g * torch.log(g) + (1.0 - g) * torch.log(1.0 - g)).mean()
            return keep_loss + self.entropy_weight * entropy
        return keep_loss
    


class ConvX(nn.Module):
    """ Conv-bn module"""
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, groups=1, dilation=1, act='relu', layer_norm=False, rms_norm=False):
        super(ConvX, self).__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel,
                              stride=stride, padding=padding, groups=groups,
                              dilation=dilation, bias=False)
        if rms_norm:
            self.bn = nn.RMSNorm(out_planes)
        else:
            self.bn = get_norm('LN', out_planes) if layer_norm else nn.BatchNorm2d(out_planes)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        """ forward """
        out = self.act(self.bn(self.conv(x.contiguous())))
        return out


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act='silu', layer_norm=False, rms_norm=False):
        """ ch_in, ch_out, shortcut, groups, kernels, expand """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='silu', layer_norm=False, rms_norm=False):
        """ ch_in, ch_out, number, shortcut, groups, expansion """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX((2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm, rms_norm=rms_norm) for _ in range(n))

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MultiScaleProjector(nn.Module):
    """
    This module implements MultiScaleProjector in :paper:`lwdetr`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factors,
        num_blocks=3,
        layer_norm=False,
        rms_norm=False,
        survival_prob=1.0,
        force_drop_last_n_features=0,
        use_fafd=False,  # 🚀 CVPR v4.0: 是否使用频域门控特征解耦
        fafd_sparsity_weight=0.0,
        # FAFD 超参说明（尽量“少而可解释”）：
        # - fafd_alpha: 过滤强度（0=关闭，1=完全用滤波后特征）。建议 <=0.3，避免 early stage 破坏预训练特征。
        # - fafd_target_keep: 频域 gate 的平均“保留比例”目标（越大越保守）。建议 0.7~0.9 起跑。
        # - fafd_entropy_weight: 可选的“二值化”正则（默认关闭，避免引入额外不稳定）。
        fafd_alpha: float = 0.15,
        fafd_target_keep: float = 0.8,
        fafd_entropy_weight: float = 0.0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            use_fafd (bool): CVPR v4.0 - 是否在每个输入特征上应用频域门控
        """
        super(MultiScaleProjector, self).__init__()

        self.scale_factors = scale_factors
        self.survival_prob = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features
        self.use_fafd = use_fafd
        self.fafd_sparsity_weight = fafd_sparsity_weight
        self.fafd_alpha = float(fafd_alpha)
        self.fafd_target_keep = float(fafd_target_keep)
        self.fafd_entropy_weight = float(fafd_entropy_weight)
        
        # 🚀 CVPR v4.0: 为每个输入特征层级添加FAFD模块
        if self.use_fafd:
            self.fafd_layers = nn.ModuleList([
                FrequencyGatedBlock(
                    in_ch,
                    reduction=4,
                    alpha=self.fafd_alpha,
                    target_keep=self.fafd_target_keep,
                    entropy_weight=self.fafd_entropy_weight,
                )
                for in_ch in in_channels
            ])

        stages_sampling = []
        stages = []
        # use_bias = norm == ""
        use_bias = False
        self.use_extra_pool = False
        for scale in scale_factors:
            stages_sampling.append([])
            for in_dim in in_channels:
                out_dim = in_dim
                layers = []

                # if in_dim > 512:
                #     layers.append(ConvX(in_dim, in_dim // 2, kernel=1))
                #     in_dim = in_dim // 2

                if scale == 4.0:
                    layers.extend([
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        get_norm('LN', in_dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    ])
                    out_dim = in_dim // 4
                elif scale == 2.0:
                    # a hack to reduce the FLOPs and Params when the dimention of output feature is too large
                    # if in_dim > 512:
                    #     layers = [
                    #         ConvX(in_dim, in_dim // 2, kernel=1),
                    #         nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    #     ]
                    #     out_dim = in_dim // 4
                    # else:
                    layers.extend([
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                    ])
                    out_dim = in_dim // 2
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.extend([
                        ConvX(in_dim, in_dim, 3, 2, layer_norm=layer_norm),
                    ])
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError("Unsupported scale_factor:{}".format(scale))
                layers = nn.Sequential(*layers)
                stages_sampling[-1].append(layers)
            stages_sampling[-1] = nn.ModuleList(stages_sampling[-1])

            in_dim = int(sum(in_channel // max(1, scale) for in_channel in in_channels))
            layers = [
                C2f(in_dim, out_channels, num_blocks, layer_norm=layer_norm),
                get_norm('LN', out_channels),
            ]
            layers = nn.Sequential(*layers)
            stages.append(layers)

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # 🚀 CVPR v4.0: 在处理前对每个特征层级应用FAFD
        if self.use_fafd:
            x = [self.fafd_layers[j](feat) for j, feat in enumerate(x)]
        
        num_features = len(x)
        if self.survival_prob < 1.0 and self.training:
            final_drop_prob = 1 - self.survival_prob
            drop_p = np.random.uniform()
            for i in range(1, num_features):
                critical_drop_prob = i * (final_drop_prob / (num_features - 1))
                if drop_p < critical_drop_prob:
                    x[i][:] = 0
        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                # don't do it inplace to ensure the compiler can optimize out the backbone layers
                x[-(i+1)] = torch.zeros_like(x[-(i+1)])
                
        results = []
        # x list of len(out_features_indexes)
        for i, stage in enumerate(self.stages):
            feat_fuse = []
            for j, stage_sampling in enumerate(self.stages_sampling[i]):
                feat_fuse.append(stage_sampling(x[j]))
            if len(feat_fuse) > 1:
                feat_fuse = torch.cat(feat_fuse, dim=1)
            else:
                feat_fuse = feat_fuse[0]
            results.append(stage(feat_fuse))
        if self.use_extra_pool:
            results.append(
                F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0)
            )
        return results

    def get_fafd_gate_stats(self):
        if not self.use_fafd:
            return None
        stats = [layer.get_gate_stats() for layer in self.fafd_layers if layer.get_gate_stats() is not None]
        if not stats:
            return None
        mean = sum(s["mean"] for s in stats) / len(stats)
        std = sum(s["std"] for s in stats) / len(stats)
        sparsity = sum(s["sparsity"] for s in stats) / len(stats)
        return {"mean": mean, "std": std, "sparsity": sparsity, "layers": stats}

    def get_fafd_sparsity_loss(self):
        # Return the *raw* sparsity loss (unweighted).
        # Weighting is handled centrally by the criterion's weight_dict
        # (e.g., 'loss_fafd_sparsity': args.fafd_sparsity_weight).
        if not self.use_fafd:
            return None
        losses = [layer.get_sparsity_loss() for layer in self.fafd_layers if layer.get_sparsity_loss() is not None]
        if not losses:
            return None
        return torch.stack(losses).mean()

    def get_fafd_gate_visual(self):
        if not self.use_fafd:
            return None
        for layer in self.fafd_layers:
            gate = getattr(layer, "_last_gate", None)
            if gate is not None:
                return gate
        return None


class SimpleProjector(nn.Module):
    def __init__(self, in_dim, out_dim, factor_kernel=False):
        super(SimpleProjector, self).__init__()
        if not factor_kernel:
            self.convx1 = ConvX(in_dim, in_dim*2, layer_norm=True, act='silu')
            self.convx2 = ConvX(in_dim*2, out_dim, layer_norm=True, act='silu')
        else:
            self.convx1 = ConvX(in_dim, out_dim, kernel=(3, 1), layer_norm=True, act='silu')
            self.convx2 = ConvX(out_dim, out_dim, kernel=(1, 3), layer_norm=True, act='silu')
        self.ln = get_norm('LN', out_dim)

    def forward(self, x):
        """ forward """
        out = self.ln(self.convx2(self.convx1(x[0])))
        return [out]
