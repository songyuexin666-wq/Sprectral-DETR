# Spectral-DETR: Robust Object Detection in Degraded Mine Scenes

**Spectral-DETR** 是在 [RF-DETR](https://github.com/roboflow/rf-detr) 基础上扩展的目标检测框架，面向 **矿井下粉尘、模糊、低照度** 等严重退化成像场景。  
通过 **频域特征净化（FAFD）**、**Query 对比去噪（QCD）** 和 **定位不确定性估计（LUE）** 三个模块，在不依赖独立图像增强网络的前提下，提升 DETR 类检测器的鲁棒性与可解释性。

- 代码仓库（Spectral-DETR）：`https://github.com/songyuexin666-wq/Sprectral-DETR`  
- 自建数据集（Mine-Objects）：`https://github.com/songyuexin666-wq/mine-datasets`

---

## 1. 项目简介

在实际矿井监控场景中，存在以下典型退化因素：

- 粉尘 / 烟雾导致的高频噪声；
- 照度不足和强对比度变化；
- 车载摄像头引入的运动模糊和抖动。

传统 DETR/RF-DETR 在这些场景下容易出现：

- 特征被噪声淹没、注意力失效；
- 解码器 Query 前景/背景混淆；
- 边界模糊样本上定位梯度不稳定。

**Spectral-DETR** 针对上述问题，从 **特征、Query 和定位** 三个阶段同时改进 RF-DETR：

- **FAFD（Frequency-Aware Feature Disentanglement）**  
  在 Transformer 编码前对多尺度特征做 2D FFT；利用退化条件驱动的频域门控削弱噪声主导频段，保留结构语义，再通过残差方式与原特征融合，减轻谱混叠与伪边缘。

- **QCD（Query Contrastive Denoising）**  
  在解码阶段对 decoder query 施加监督对比约束：  
  以最终层匹配 Query 为锚点，跨层同一 Query 为正样本，歧义背景 Query 为难负样本，提升前景/背景可分性。仅在训练时启用，对推理无额外开销。

- **LUE（Localization Uncertainty Estimation）**  
  对每个框预测坐标均值与 log-variance，用不确定性自适应加权定位梯度，并对预测不确定性与真实误差做校准。输出可解释的“每框可靠性”指标，便于安全场景下的人工复核与告警策略。

---

## 2. 数据集说明

### 2.1 自建 Mine-Objects 数据集（3081 张，14 类）

仓库地址：[`mine-datasets`](https://github.com/songyuexin666-wq/mine-datasets)  

- **场景**：真实地下矿井巷道，车载防爆摄像头采集，存在明显的昏暗、模糊、粉尘等退化因素。
- **数量**：3,081 张高分辨率图像，按 `train / val / test` 划分。
- **类别数**：14 个矿井相关类别：

  1. `person` – 井下人员  
  2. `redlight` – 红色信号灯 / 报警灯  
  3. `light` – 普通照明灯  
  4. `port` – 接口 / 端口（电缆接口等）  
  5. `sign` – 标志牌 / 警示牌  
  6. `warn` – 其他警告设施  
  7. `gear` – 齿轮 / 机械转动部件  
  8. `car` – 车辆（非专用矿车）  
  9. `mine-car` – 矿车 / 轨道车  
  10. `ele-warn` – 电气警示装置  
  11. `camera` – 监控摄像头  
  12. `generator` – 发电机 / 电力设备  
  13. `annihilator` – 灭火器 / 灭火装置  
  14. `electric-wire` – 电缆 / 电线  

- **标注格式**：提供适配 COCO/YOLO 的标注文件，可直接用于 Spectral-DETR 与常见检测器。

> 更详细的数据集结构与使用方式，请见数据集仓库 [`mine-datasets`](https://github.com/songyuexin666-wq/mine-datasets) 的 README。

### 2.2 ScienceDB Mine 数据集（跨域评估）

- 平台入口：`https://www.scidb.cn/`（ScienceDB，中国科学院科学数据库）  
- 使用方式：在 ScienceDB 平台上根据论文中的引用信息或“mine / underground / coal mine”等关键字检索相应矿山场景数据集。  
- 用途：作为 **跨域矿山场景**，评估 Spectral-DETR 在不同传感器与地质条件下的泛化能力。

> 在正式论文中建议给出具体数据集 DOI 或 ScienceDB 数据集页面链接，这里在 README 中提供平台入口，方便读者自行检索。

### 2.3 ExDark 数据集（低照度场景）

- 官方 GitHub：`https://github.com/cs-chan/Exclusively-Dark-Image-Dataset`  
- 特点：7,363 张极低照度图像，覆盖 10 种光照条件和 12 个物体类别。  
- 用途：验证 Spectral-DETR 在 **极低信噪比夜间/暗光场景** 下的鲁棒性，与 Mine-Objects 形成互补。

---

## 3. 环境与安装

### 3.1 克隆仓库

```bash
git clone https://github.com/songyuexin666-wq/Sprectral-DETR.git
cd Sprectral-DETR
```

### 3.2 创建环境并安装依赖

建议使用 Python ≥ 3.9，CUDA 版本与 PyTorch 官方支持对应。

```bash
pip install -r requirements.txt
```

`requirements.txt` 中包含：

- 深度学习框架：`torch`, `torchvision`, `transformers`, `peft`, `supervision`
- 训练与配置：`numpy`, `pyyaml`, `pydantic`, `tensorboard`, `wandb`, `torchinfo`
- COCO 与图像处理：`pycocotools`, `Pillow`, `tqdm`, `matplotlib`, `scikit-learn`
- 可选部署组件（ONNX / TensorRT 等）

如果暂时不需要 TensorRT/ONNX 部署，可自行从 `requirements.txt` 中删除相关依赖。

---

## 4. 快速开始：在 Mine-Objects 上训练

### 4.1 准备数据集

1. 下载并解压自建数据集：[`mine-datasets`](https://github.com/songyuexin666-wq/mine-datasets)  
2. 假设你的目录结构为：

```text
/path/to/mine-datasets/
├── train/ ...
├── val/   ...
└── test/  ...
```

### 4.2 配置文件

本仓库提供两份示例配置：

- `configs/baseline.yaml`  
  - 基于 RF-DETR 的 **纯基线配置**，关闭 LUE / FAFD / QCD，用于消融对比。

- `configs/lue_fafd_qcd.yaml`  
  - 完整 Spectral-DETR 配置，**同时启用 LUE + FAFD + QCD**，并给出推荐超参数（如 `fafd_alpha=0.15`, `qcd_temperature=0.15`, `lue_warmup_epochs=15` 等）。

你只需根据实际数据路径修改配置中的 `dataset` 部分，例如：

```yaml
dataset:
  dataset_file: "coco"
  coco_path: "/path/to/mine-datasets"
```

### 4.3 启动训练

使用专门的矿井场景训练脚本 `train_mine.py`：

```bash
# 纯基线 RF-DETR（用于对比）
python3 train_mine.py --config configs/baseline.yaml

# 完整 Spectral-DETR（FAFD + QCD + LUE 全开）
python3 train_mine.py --config configs/lue_fafd_qcd.yaml
```

脚本会自动完成：

- 检查数据集目录与标注格式（COCO / Roboflow）；
- 从标注文件中解析类别数；
- 根据 `pretrain_weights` 智能选择 RF-DETR Base / Medium / Large；
- 打印模型参数量与 FLOPs；
- 将完整配置保存到输出目录，并记录诊断信息（退化桶、样例可视化等）。

---

## 5. 在其他数据集上训练（ScienceDB / ExDark）

只需在配置文件中调整 `dataset` 部分指向目标数据集，并保证标注转换为 COCO 格式：

```yaml
dataset:
  dataset_file: "coco"
  coco_path: "/path/to/sciencedb_or_exdark_coco_style"
```

然后复用同一训练脚本：

```bash
python3 train_mine.py --config configs/lue_fafd_qcd.yaml
```

> 注：ExDark 原始提供为分类/检测格式，需先转换为 COCO 标注；本仓库中可添加相应的转换脚本。

---

## 6. 推理示例（单张图像）

以下示例展示如何在 Python 中加载 Spectral-DETR 模型并对单张图像进行推理（接口与原 RF-DETR 保持一致）：

```python
import torch
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# 根据需要选择 RFDETRBase / RFDETRMedium / RFDETRLarge，并加载 Spectral-DETR 训练好的权重
model = RFDETRBase()
model.load_state_dict(torch.load("/path/to/spectral_detr_mineobjects.pth", map_location="cpu"))
model.eval()

image = Image.open("/path/to/your_image.jpg").convert("RGB")
detections = model.predict(image, threshold=0.5)

for cls_id, conf, box in zip(detections.class_id, detections.confidence, detections.bbox):
    print(COCO_CLASSES[cls_id], conf, box)
```

---

## 7. 与原 RF-DETR 的关系

- 本仓库基于 [RF-DETR](https://github.com/roboflow/rf-detr) 源码进行扩展和适配，主要增加了：
  - 矿井退化场景的数据处理与诊断工具；
  - FAFD / QCD / LUE 三个模块及其超参搜索结果；
  - 专用于 Mine-Objects / ScienceDB / ExDark 的训练脚本与配置。

- 原 RF-DETR 的特性（如实时性能、Segmentation Head、优化推理等）仍可参考其官方仓库与文档。

---

## 8. 致谢与引用

### 致谢

本工作构建在以下优秀开源项目之上：

- [RF-DETR](https://github.com/roboflow/rf-detr)
- LW-DETR
- DINOv2
- Deformable DETR

感谢这些工作的作者开放源码。

### 引用（示例）

如果本仓库或 Mine-Objects 数据集对你的研究有帮助，欢迎在论文中引用或致谢：

```text
Spectral-DETR and Mine-Objects Dataset,
https://github.com/songyuexin666-wq/Sprectral-DETR
https://github.com/songyuexin666-wq/mine-datasets
```

---

## 9. License

本仓库代码在 **MIT License** 下发布，具体条款见本仓库中的 `LICENSE` 文件。  
在使用本代码与数据集时，请遵循相应许可证要求。

