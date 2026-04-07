# 我负责的部分：Data/IO + SAM2 + Hand-crafted Baseline

这部分覆盖视频目标移除项目中由我负责的前半段工程模块。当前运行验证在 `sam3` conda 环境中完成。

## 我负责什么

- 视频预处理与标准化
- 抽帧与视频重组
- frame / mask / result 的对齐检查
- `meta.json` 元数据生成
- SAM2 mask 提取入口与 overlay 导出
- hand-crafted baseline 入口、mask 生成、`cv2.inpaint` 结果与 overlay 导出
- 公共路径、配置、日志工具

## 我不负责什么

- ProPainter 或更强的后续 inpainting 主链
- 全量 benchmark / evaluation
- Part 3 扩展内容
- 整个 repo 的重构
- teammate 负责的后续消费模块

## 标准目录契约

```text
data/<video_name>/
  raw/
    video.mp4
  processed/
    frames/
      00000.png
      00001.png
    masks_sam2/
    masks_baseline/
    overlays/
    results_baseline/
    videos/
    meta.json
```

## I/O 契约

- 帧命名统一为 5 位零填充：`00000.png`、`00001.png`
- mask 为单通道 8-bit PNG
- 背景为 `0`，前景为 `255`
- 同一个视频下，`frames`、`masks_sam2`、`masks_baseline`、`results_baseline` 尺寸必须完全一致
- overlay 视频尺寸必须与 frame 一致

## CLI 约定

主脚本统一支持：

- `--input_video`
- `--output_dir`
- `--config`
- `--device`
- `--save_overlay`

## 给 Teammate 的交接路径

后续使用 SAM2 输出的模块，直接读取：

- `data/<video_name>/processed/frames/`
- `data/<video_name>/processed/masks_sam2/`
- `data/<video_name>/processed/meta.json`

后续使用 baseline 输出的模块，直接读取：

- `data/<video_name>/processed/masks_baseline/`
- `data/<video_name>/processed/results_baseline/`
- `data/<video_name>/processed/videos/baseline_result.mp4`
- 如果开启 overlay，再读取 `data/<video_name>/processed/overlays/baseline_overlay.mp4`

## SAM2 使用说明

入口脚本：

```bash
python scripts/run_sam2_masks.py \
  --input_video data/tennis/raw/video.mp4 \
  --output_dir data/tennis \
  --config configs/sam2.yaml \
  --device cuda \
  --save_overlay
```

prompt 初始化来自 `sam2.prompt_json`，JSON 结构应为：

```json
{
  "video_name": "tennis",
  "prompts": [
    {
      "frame_index": 0,
      "bbox": [x1, y1, x2, y2],
      "points": [
        {"x": 120, "y": 220, "label": 1}
      ]
    }
  ]
}
```

说明：

- `frame_index` 为 0-based
- `bbox` 可选，但首版已支持
- `points` 可选，可与 `bbox` 一起使用
- `label` 采用正负点约定
- 如果缺少 `prompt_json`，且配置未开启自动模式，脚本会直接报错
- `masks_sam2/*.png` 必须保持为 `0/255` 二值 mask，且与对应 frame 同尺寸

## Hand-crafted Baseline 使用说明

入口脚本：

```bash
python scripts/run_handcrafted_baseline.py \
  --input_video data/tennis/raw/video.mp4 \
  --output_dir data/tennis \
  --config configs/handcrafted.yaml \
  --device cuda \
  --save_overlay
```

预期输出：

- `processed/masks_baseline/00000.png`、`00001.png` ...
- `processed/results_baseline/00000.png`、`00001.png` ...
- `processed/videos/baseline_result.mp4`
- 可选 `processed/overlays/baseline_overlay.mp4`
- `processed/meta.json` 继续作为统一元数据契约

baseline mask 约束：

- 二值 `0/255` PNG
- 与对应 frame 同宽高
- 每帧一个 mask

配置说明：

- `baseline.detector_backend`：候选目标生成后端，例如 `yolov8_seg`、`mask_rcnn`
- `baseline.detector_weights`：检测器权重路径
- `baseline.target_classes`：保留的动态类
- `baseline.use_lk_motion_filter`：是否启用稀疏 LK 光流筛动态
- `baseline.use_temporal_borrowing`：预留的后续增强
- `motion.*`：LK 光流窗口、金字塔层数、终止条件、运动阈值
- `mask_refine.*`：形态学和连通域清理参数
- `inpaint.method`：OpenCV inpainting 方法，默认 `telea`
- `inpaint.radius`：inpainting 半径
- `output.*`：baseline 输出目录名
- `baseline.device` 和 `baseline.save_overlay` 与 CLI 对应
- `masks_baseline/*.png` 必须保持为 `0/255` 二值 mask，且与对应 frame 同尺寸

## 已实现脚本

- `scripts/prepare_video.py`
- `scripts/extract_frames.py`
- `scripts/compose_video.py`
- `scripts/check_alignment.py`
- `scripts/run_sam2_masks.py`
- `scripts/run_handcrafted_baseline.py`

## 快速开始

先把输入视频标准化：

```bash
python scripts/prepare_video.py \
  --input_video path/to/input.mp4 \
  --output_dir data/tennis \
  --target_width 854 \
  --target_height 480 \
  --keep_aspect
```

抽帧并生成 `meta.json`：

```bash
python scripts/extract_frames.py \
  --input_video data/tennis/raw/video.mp4 \
  --output_dir data/tennis
```

把帧重组回视频：

```bash
python scripts/compose_video.py \
  --output_dir data/tennis
```

检查 frame 和 mask 是否对齐：

```bash
python scripts/check_alignment.py \
  --output_dir data/tennis \
  --mask_dir data/tennis/processed/masks_sam2 \
  --expect_binary_masks
```

## 验证方式

SAM2 输出检查：

```bash
python scripts/check_alignment.py \
  --output_dir data/tennis \
  --mask_dir data/tennis/processed/masks_sam2 \
  --expect_binary_masks
```

Baseline 输出检查：

```bash
python scripts/check_alignment.py \
  --output_dir data/tennis \
  --mask_dir data/tennis/processed/masks_baseline \
  --expect_binary_masks
```

还应检查：

- `processed/meta.json` 是否存在，且帧数、分辨率正确
- `processed/overlays/sam2_overlay.mp4` 是否生成
- `processed/videos/baseline_result.mp4` 是否生成

## 最终状态

你负责的这部分已经到达可交接状态。

- 阶段 1 已完成：
  - 数据标准化
  - 抽帧
  - 视频重组
  - 对齐检查
  - `meta.json`
- 阶段 2 已完成：
  - `scripts/run_sam2_masks.py`
  - `src/segmentation/sam2_pipeline.py`
  - `src/mask_refine/postprocess.py`
  - `src/visualization/overlay.py`
  - 已在 `sam3` 中完成真实验证
- 阶段 3 已完成：
  - `scripts/run_handcrafted_baseline.py`
  - `src/detection/detector.py`
  - `src/motion/lk_motion_filter.py`
  - `src/inpainting/baseline_inpaint.py`
  - 已在 `sam3` 中完成 smoke 验证
- 阶段 4 已完成：
  - 补了输入和路径检查
  - 补了更明确的日志
  - 补了 teammate handoff 说明
  - 主运行脚本已支持相对路径配置解析

## 最终验证总结

基于本地 smoke case，已确认：

- `prepare_video.py`、`extract_frames.py`、`compose_video.py`、`check_alignment.py` 正常运行
- `run_sam2_masks.py` 成功生成：
  - `processed/masks_sam2/`
  - `processed/overlays/sam2_overlay.mp4`
- `run_handcrafted_baseline.py` 成功生成：
  - `processed/masks_baseline/`
  - `processed/results_baseline/`
  - `processed/videos/baseline_result.mp4`
  - `processed/overlays/baseline_overlay.mp4`
- smoke case 上的帧数、尺寸一致性、二值 mask 检查均通过

## 已知限制

- SAM2 当前是在 `sam3` 中以 CPU 路径完成验证
- 本机上 SAM2 的可选 `_C` 扩展没有成功加载，因此部分内部后处理会被跳过
- `automatic_mode` 预留但尚未实现
- baseline 的 `use_temporal_borrowing` 预留但尚未实现
- `yolov8_seg` 只完成了接口预留，如要启用仍需在当前环境中安装 `ultralytics`

## 提交前说明

- 建议保留源码、配置文件和本 README
- 提交前自行决定是否保留 smoke 专用内容：
  - `configs/sam2_smoke.yaml`
  - `configs/handcrafted_smoke.yaml`
  - `data/smoke_case/`
  - `tmp_smoke/`
  - `tmp_sam2_smoke/`
- 对 teammate 的稳定契约应保持不变：
  - `frames/`
  - `masks_sam2/`
  - `masks_baseline/`
  - `results_baseline/`
  - `videos/`
  - `meta.json`
