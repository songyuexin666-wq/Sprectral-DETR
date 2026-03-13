# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

"""
自适应超参数管理器 - 解决超参数过多问题

核心思想：
1. 三大创新点协同：FAFD → QCD → LUE 相互调节
2. 自适应权重：根据训练阶段和数据统计自动调整
3. 简化配置：用户只需配置少量核心参数

Author: Claude + 宋跃鑫
Date: 2026-02-06
"""

import torch
import torch.nn as nn


class AdaptiveParamsManager(nn.Module):
    """
    自适应超参数管理器

    三大创新点协同机制：
    1. FAFD的频域稀疏度 → 指导LUE的初始不确定性
    2. QCD的query相似度 → 调整对比学习温度
    3. LUE的不确定性 → 调整QCD的对比学习权重
    """

    def __init__(
        self,
        # 核心超参（用户配置）
        use_lue: bool = True,
        use_fafd: bool = True,
        use_qcd: bool = True,
        # 简化配置：只需配置这3个主要参数
        innovation_strength: float = 1.0,  # 创新点整体强度 [0.5-1.5]
        adaptation_rate: float = 0.1,      # 自适应调整速率 [0.05-0.2]
        warmup_epochs: int = 5,            # 统一warm-up周期
        # QCD 配置（从 config yaml 传入，覆盖内部默认值）
        qcd_base_weight: float = 0.1,          # 对比学习基础权重
        qcd_initial_temperature: float = 0.07, # 温度初始值（可学习）
        qcd_hard_negatives_k: int = 50,        # hard negative 数量
    ):
        super().__init__()

        self.use_lue = use_lue
        self.use_fafd = use_fafd
        self.use_qcd = use_qcd

        # 核心超参
        self.innovation_strength = innovation_strength
        self.adaptation_rate = adaptation_rate
        self.warmup_epochs = warmup_epochs
        self.qcd_base_weight = qcd_base_weight
        self.qcd_hard_negatives_k = qcd_hard_negatives_k

        # 可学习的权重（自动优化）
        if use_lue:
            # LUE: 坐标解耦权重
            self.lue_center_weight = nn.Parameter(torch.ones(2))
            self.lue_size_weight = nn.Parameter(torch.ones(2) * 0.8)
            # LUE: 方差正则权重
            self.lue_var_reg_weight = nn.Parameter(torch.tensor(0.1))

        if use_fafd:
            # FAFD: 频域混合比例
            self.fafd_alpha = nn.Parameter(torch.tensor(0.15))

        if use_qcd:
            # QCD: 对比学习温度（可学习），从 config 传入初始值
            self.qcd_temperature = nn.Parameter(torch.tensor(qcd_initial_temperature))

        # 统计缓存（用于自适应调整）
        self.register_buffer('_fafd_sparsity', torch.tensor(0.5))
        self.register_buffer('_qcd_similarity_std', torch.tensor(0.1))
        self.register_buffer('_lue_uncertainty_mean', torch.tensor(0.0))

        # Epoch计数
        self.register_buffer('_epoch', torch.tensor(0))

    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self._epoch.fill_(epoch)

    def get_warmup_ratio(self) -> float:
        """计算warm-up比例"""
        if self.warmup_epochs == 0:
            return 1.0
        return min(1.0, float(self._epoch) / self.warmup_epochs)

    def update_statistics(
        self,
        fafd_sparsity: torch.Tensor = None,
        qcd_similarity_std: torch.Tensor = None,
        lue_uncertainty_mean: torch.Tensor = None
    ):
        """
        更新统计信息（用于协同调整）

        Args:
            fafd_sparsity: FAFD频域稀疏度
            qcd_similarity_std: QCD query相似度标准差
            lue_uncertainty_mean: LUE平均不确定性
        """
        # 使用指数移动平均更新统计
        alpha = self.adaptation_rate

        if fafd_sparsity is not None:
            self._fafd_sparsity = (1 - alpha) * self._fafd_sparsity + alpha * fafd_sparsity.detach()

        if qcd_similarity_std is not None:
            self._qcd_similarity_std = (1 - alpha) * self._qcd_similarity_std + alpha * qcd_similarity_std.detach()

        if lue_uncertainty_mean is not None:
            self._lue_uncertainty_mean = (1 - alpha) * self._lue_uncertainty_mean + alpha * lue_uncertainty_mean.detach()

    def get_lue_params(self):
        """
        获取LUE参数（自适应调整）

        协同机制：
        - FAFD稀疏度高 → 特征质量好 → 降低初始不确定性
        - QCD相似度分散 → query区分度高 → 可以提高不确定性权重
        """
        if not self.use_lue:
            return None

        # 基础权重
        uncertainty_weight = 0.5 * self.innovation_strength

        # 协同调整1: FAFD稀疏度 → 不确定性权重
        if self.use_fafd:
            # 稀疏度高（0.8）→ 特征质量好 → 提升不确定性权重
            # 稀疏度低（0.2）→ 特征质量差 → 降低不确定性权重
            fafd_factor = 0.5 + float(self._fafd_sparsity)
            uncertainty_weight *= fafd_factor

        # 协同调整2: QCD相似度 → 不确定性权重
        if self.use_qcd:
            # 相似度分散（std大）→ query区分度高 → 提升不确定性权重
            qcd_factor = 0.8 + 2.0 * float(self._qcd_similarity_std)
            uncertainty_weight *= torch.clamp(torch.tensor(qcd_factor), 0.5, 1.5).item()

        return {
            'uncertainty_weight': uncertainty_weight,
            'center_weight': self.lue_center_weight,
            'size_weight': self.lue_size_weight,
            'var_reg_weight': self.lue_var_reg_weight,
            'warmup_ratio': self.get_warmup_ratio()
        }

    def get_fafd_params(self):
        """
        获取FAFD参数（自适应调整）

        协同机制：
        - LUE不确定性高 → 图像质量差 → 增强频域滤波
        """
        if not self.use_fafd:
            return None

        # 基础混合比例
        alpha = float(self.fafd_alpha)

        # 协同调整: LUE不确定性 → FAFD滤波强度
        if self.use_lue:
            # 不确定性高 → 增强滤波（提高alpha）
            unc_factor = 1.0 + 0.5 * float(self._lue_uncertainty_mean)
            alpha *= unc_factor
            alpha = min(alpha, 0.5)  # 限制最大值

        return {
            'alpha': alpha,
            'target_keep': 0.8,  # 目标保留比例
            'entropy_weight': 0.0  # 熵正则权重
        }

    def get_qcd_params(self):
        """
        获取QCD参数（自适应调整）

        协同机制：
        - LUE不确定性高 → 降低对比学习权重（避免过拟合）
        - FAFD稀疏度低 → 特征质量差 → 降低温度（增强对比）
        """
        if not self.use_qcd:
            return None

        # 基础权重（来自 config，支持 innovation_strength 缩放）
        qcd_weight = self.qcd_base_weight * self.innovation_strength
        temperature = float(self.qcd_temperature)

        # 协同调整1: LUE不确定性 → QCD权重
        if self.use_lue:
            # 不确定性高 → 降低对比学习权重
            unc_factor = 1.0 - 0.3 * float(self._lue_uncertainty_mean)
            qcd_weight *= max(unc_factor, 0.5)

        # 协同调整2: FAFD稀疏度 → QCD温度
        if self.use_fafd:
            # 稀疏度低 → 特征质量差 → 降低温度（增强对比）
            temp_factor = 0.7 + 0.6 * float(self._fafd_sparsity)
            temperature *= torch.clamp(torch.tensor(temp_factor), 0.5, 1.2).item()

        return {
            'weight': qcd_weight,
            'temperature': temperature,
            'hard_negatives_k': self.qcd_hard_negatives_k,  # 来自 config
            'neg_pos_ratio': 10  # 负正样本比例
        }

    def get_all_params(self):
        """获取所有参数（用于日志记录）"""
        params = {
            'warmup_ratio': self.get_warmup_ratio(),
            'epoch': int(self._epoch)
        }

        if self.use_lue:
            params['lue'] = self.get_lue_params()

        if self.use_fafd:
            params['fafd'] = self.get_fafd_params()

        if self.use_qcd:
            params['qcd'] = self.get_qcd_params()

        # 统计信息
        params['statistics'] = {
            'fafd_sparsity': float(self._fafd_sparsity),
            'qcd_similarity_std': float(self._qcd_similarity_std),
            'lue_uncertainty_mean': float(self._lue_uncertainty_mean)
        }

        return params

    def __repr__(self):
        s = f"AdaptiveParamsManager(\n"
        s += f"  innovation_strength={self.innovation_strength},\n"
        s += f"  adaptation_rate={self.adaptation_rate},\n"
        s += f"  warmup_epochs={self.warmup_epochs},\n"
        s += f"  use_lue={self.use_lue}, use_fafd={self.use_fafd}, use_qcd={self.use_qcd}\n"
        s += f")"
        return s
