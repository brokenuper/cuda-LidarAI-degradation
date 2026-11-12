# BEVFusion 冻结骨干微调（Frozen-Backbone Fine-tuning）

> 本文档记录了针对 **RTX 3080 10G/12G 显存、算力友好型冻结微调** 的全部实施工作与可复现操作。
> 冻结 camera（ResNet50）与 lidar（SparseEncoder）两个 backbone，只开放 **Fuser + Decoder + Detection Head** 的梯度。
> 数据集：nuScenes-mini；环境：统一 conda 环境 `.conda-bevfusion-eval`。

---

## 1. 目标与动机

原始训练会在 10G 卡上 OOM，且整网训练算力消耗大。本方案通过“冻结骨干、只训融合与检测头”，把可训练参数量从 **36.61M 降到 10.16M（27.8%）**，同时由于冻结部分不参与反向图，显存与算力都大幅下降：

| 部件 | 是否冻结 | 参数量 |
|---|---|---|
| `encoders.camera.backbone`（ResNet50） | ✅ 冻结（eval + requires_grad=False） | 23.51M |
| `encoders.lidar.backbone`（SparseEncoder） | ✅ 冻结（eval + requires_grad=False） | 2.69M |
| `encoders.camera.neck / vtransform` | ❌ 可训练 | — |
| `fuser`（ConvFuser，BEV 融合） | ❌ 可训练 | — |
| `decoder`（BEV backbone + neck） | ❌ 可训练 | — |
| `heads.object`（TransFusion 检测头） | ❌ 可训练 | — |
| **合计** | 冻结 26.20M / 总 36.61M | 可训练 10.16M |

---

## 2. 实现原理与关键坑

### 2.1 冻结 = eval() + requires_grad=False
- **`requires_grad=False`**：优化器只接受 `requires_grad=True` 的参数（`build_optimizer` 天然过滤），且 autograd 不会为冻结部分保留中间激活，从而省显存。
- **`.eval()`**：冻结部分的 BN 统计（running mean/var）与 Dropout 保持固定、不漂移。

### 2.2 坑 1：runner 每轮会把模型切回 train 态 ⚠️
`mmcv.EpochBasedRunner.train()` 在每个 epoch 开头都会执行 `self.model.train()`，这会把已 `.eval()` 的冻结 backbone 又切回训练态（其 BN 统计会继续更新）。
**对策**：注册自定义 hook `FrozenBackboneEvalHook`（在 `before_train_epoch` 时对冻结模块重新 `.eval()`），已在 `mmdet3d/apis/train.py` 注册。

### 2.3 坑 2：配置键名不要用 `freeze` ⚠️
`mmcv.utils.config.ConfigDict` 自带 `freeze()` **方法**，属性访问 `cfg.freeze` 会拿到绑定方法而不是配置列表。
**对策**：配置里使用键名 **`freeze_modules`**。

### 2.4 坑 3：`ConvFuser` 不会跟随 recursive 加载自动出现
`convfuser.yaml` 与训练入口是**兄弟**配置文件，`torchpack configs.load(..., recursive=True)` 只递归合并父目录的 `default.yaml`，**不会**合并兄弟文件 → `model.fuser` 会是 `None`。
**对策**：`model.fuser` 的完整定义已**内联**进 `convfuser_frozenfinetune.yaml`。

### 2.5 坑 4：mini 数据缺两样东西，必须裁剪训练 pipeline
- **无 `nuscenes_dbinfos_train.pkl`** → 训练 pipeline 不能有 `ObjectPaste`。
- **无 `maps/expansion/*.json`** → 训练 pipeline 不能有 `LoadBEVSegmentation`，`Collect3D.key` 也要去掉 `gt_masks_bev`。

---

## 3. 代码改动清单

| 文件 | 类型 | 内容 |
|---|---|---|
| `bevfusion/mmdet3d/utils/freeze.py` | 新增 | `freeze_model()` + `FrozenBackboneEvalHook`（注册进 mmcv `HOOKS`） |
| `bevfusion/mmdet3d/utils/__init__.py` | 修改 | 导出 `freeze_model` / `FrozenBackboneEvalHook` |
| `bevfusion/mmdet3d/apis/train.py` | 修改 | runner 构建后按 `cfg.freeze_modules` 注册 `FrozenBackboneEvalHook` |
| `bevfusion/tools/train.py` | 修改 | `build_model + init_weights(+sync_bn)` 之后调用 `freeze_model(model, cfg.freeze_modules)`，并打印可训练参数 |
| `bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser_frozenfinetune.yaml` | 新增 | mini 冻结微调完整配置 |

---

## 4. 配置文件要点（`convfuser_frozenfinetune.yaml`）

- `freeze_modules: [encoders.camera.backbone, encoders.lidar.backbone]`
- `load_from: <CUDA-BEVFusion>/model/resnet50/bevfusion-det.pth`（官方 FP32 检测权重）
- `dataset_root: /home/scut/CX/Lidar_AI_Solution/dataset/nuScenes-mini/`
- `image_size: [128, 352]`（256×704 训练会 GPU OOM；128×352 兼容预训练几何且省显存）
- `model.fuser`（ConvFuser 内联定义，见坑 3）
- `model.encoders.lidar.voxelize.max_voxels: [60000, 80000]`（见坑 5，治 GPU OOM）
- `data.samples_per_gpu: 1; data.workers_per_gpu: 1`（见坑 6，治主机内存 OOM）
- `train_pipeline`：去掉 `ObjectPaste` / `LoadBEVSegmentation` / `gt_masks_bev`（见坑 4）
- `log_config`：只留 `TextLoggerHook`（绕开 `torch.utils.tensorboard` 在该环境的 distutils 崩溃）

### 坑 5：GPU 显存 OOM（`cuda error 2`，位于 lidar 稀疏卷积）
fp16 训练首次前向在 `SparseEncoder` 第一个稀疏卷积的 `get_indice_pairs` 报 `cuda error 2`（显存不足）。该处分配随 `max_voxels` 上限缩放；mini 场景点云远小于全量数据，把 `max_voxels` 从 `[120000, 160000]` 降到 `[60000, 80000]` 后**不再 GPU OOM**，几何不变。

### 坑 6：主机内存 OOM（`exit code 137`，Linux OOM Killer）
训练中被 OS 以 `Out of memory: Killed process ... (python)` 杀掉（`dmesg` 可查）。原因是**宿主机被其他并行 ML 任务占用**（如 char5llm/RTDETR/auxdtr 等，可用内存仅 ~5.6Gi），而非 GPU/代码问题。
对策：`workers_per_gpu: 1` 减少 loader 进程内存；**并在服务器空闲时再运行**（见 §6）。

---

## 5. 环境准备（一次性）

```bash
# 1) 激活统一 conda 环境（内含 TensorRT 8.6.1.6 等）
source /home/scut/anaconda3/etc/profile.d/conda.sh
conda activate /home/scut/CX/Lidar_AI_Solution/.conda-bevfusion-eval

# 2) 进入代码目录并加载编译/运行环境
cd /home/scut/CX/Lidar_AI_Solution/CUDA-BEVFusion/bevfusion
source ../tool/environment.sh
```

> `environment.sh` 已统一指向 `.conda-bevfusion-eval/opt/TensorRT-8.6.1.6`，并设置 `CUDA_HOME=/usr/local/cuda-11.8`、`PYTHONNOUSERSITE=1` 等。

---

## 6. 复现命令

### 6.1 冻结状态自检（不训练，几十秒）
先确认配置解析、模型构建、冻结效果是否符合预期（应看到 **total 36.61M / trainable 10.16M (27.8%)**，两个 backbone `training=False trainable=False`，fuser/decoder/heads `trainable=True`）：

```bash
python - <<'PY'
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval, freeze_model
from mmdet3d.models import build_model

configs.load('configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser_frozenfinetune.yaml', recursive=True)
cfg = Config(recursive_eval(configs))
model = build_model(cfg.model); model.init_weights()
freeze_model(model, cfg.get('freeze_modules'))
tot = sum(p.numel() for p in model.parameters())
trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'total={tot/1e6:.2f}M trainable={trn/1e6:.2f}M ({100*trn/tot:.1f}%)')
print('camera.backbone training/trainable:', model.encoders.camera.backbone.training,
      any(p.requires_grad for p in model.encoders.camera.backbone.parameters()))
print('lidar.backbone  training/trainable:', model.encoders.lidar.backbone.training,
      any(p.requires_grad for p in model.encoders.lidar.backbone.parameters()))
print('fuser trainable :', any(p.requires_grad for p in model.fuser.parameters()))
print('decoder trainable:', any(p.requires_grad for p in model.decoder.parameters()))
print('heads  trainable :', any(p.requires_grad for p in model.heads.object.parameters()))
PY
```

### 6.2 正式微调（在服务器空闲时运行）
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH="$PWD/qat:$PYTHONPATH"

torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser_frozenfinetune.yaml \
  --run-dir ../qat/ckpt/frozen_finetune_mini
```

- 从官方检测权重 `bevfusion-det.pth` 继续微调（`load_from` 已在配置中指定）。
- 训练过程中日志会打印 `[freeze] ...` 与 `Frozen backbones; trainable params: 10.16 M`，确认冻结生效。
- 每 50 步打印一次 loss；每个 epoch 结束后自动在 mini val 上评估。

### 6.3 可调参数（命令行覆盖，无需改文件）
```bash
# 例如只跑 3 个 epoch、把学习率调低
torchpack dist-run -np 1 python tools/train.py \
  configs/.../convfuser_frozenfinetune.yaml \
  --run-dir ../qat/ckpt/frozen_finetune_mini \
  max_epochs=3 optimizer.lr=1.0e-4
```

---

## 7. 已验证结果（截至 2026-11-09）

### 7.1 静态校验 ✅
- 配置可解析；`image_size=[128,352]`、`feature_size=[16,44]`、`max_epochs=5`、`freeze_modules` 正确。
- 训练 pipeline 已去掉 `ObjectPaste`/`LoadBEVSegmentation`（类型列表已核对）。
- `train dataset = CBGSDataset`，`ann_file=nuscenes_infos_train.pkl`（mini 路径）。
- 冻结后：`total 36.61M / trainable 10.16M (27.8%)`；两个 backbone `training=False, trainable=False`；neck/vtransform/fuser/decoder/heads 均 `trainable=True`。

### 7.2 冒烟训练 ✅（受主机内存所限，未跑完整 epoch）
- `[freeze]` 摘要正确输出；`FrozenBackboneEvalHook` 出现在 `before_train_epoch` hook 列表中。
- 官方 checkpoint 加载成功（仅 `vtransform.frustum` 尺寸不匹配的**预期**警告：预训练为 256×704、当前为 128×352，forward 会按当前尺寸重算，可忽略）。
- 训练进入迭代后 **GPU 峰值约 8.2GB / 10GB，稳定未 OOM**（`max_voxels` 减半生效）。
- 进程最终以 **exit 137（Linux OOM Killer）** 结束，原因见坑 6：宿主内存不足，非代码/GPU 问题。

---

## 8. 待办 / 下一步

- [ ] **服务器空闲后**执行 §6.2 的正式命令，跑完整微调。
- [ ] 记录微调前后 mini val 指标并对比基线（此前 128×352 推理基线：mAP **0.5327**、NDS **0.5619**）。
- [ ] 如需续训：把 `load_from` 改为 `qat/ckpt/frozen_finetune_mini/epoch_*.pth`，或改用 `resume_from`。
- [ ] （可选）微调完成后衔接既有 PTQ/ONNX 导出链路：`qat/ptq.py --checkpoint ../qat/ckpt/frozen_finetune_mini/...`。