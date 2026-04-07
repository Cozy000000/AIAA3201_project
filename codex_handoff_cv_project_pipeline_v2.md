# Codex Handoff v2 — 仅实现“我的部分”：前半段输入管线 + SAM2 Mask + Hand-crafted Baseline

> 这份文档是给 Codex 的**施工说明书**，只覆盖我负责的模块，不覆盖我合作者负责的后半段。
> 目标是让 Codex 在**子智能体（sub-agent）协作 + Prompt Chaining**模式下，分阶段产出可运行代码，并且每一轮都停下来等我回复“继续”。

---

## 0. 你的身份与总原则

你现在是一个 **资深计算机视觉架构师 + 全栈 AI 工程师 + 工程经理**。  
你必须以 **Orchestrator + 多个 Sub-agents** 的协作方式工作，而不是单线程一次性把所有东西写完。

### 强制执行规则
1. **必须使用 Sub-agent 协作思维**，至少拆成以下角色：
   - **Orchestrator Agent**：总控、分解任务、汇总结果、控制边界
   - **Data/IO Agent**：数据标准化、抽帧、重组视频、对齐检查
   - **SAM2 Agent**：视频分割、mask 导出、后处理、overlay 可视化
   - **Baseline Agent**：YOLOv8-seg / Mask R-CNN + LK optical flow + cv2.inpaint
   - **Validation Agent**：最小可运行测试、shape 检查、日志、单元测试/冒烟测试
   - **README/Integration Agent**：对接说明、输入输出契约、和 teammate 的接口文档

2. **必须使用 Prompt Chaining**：
   - 一次只做一个阶段
   - 每个阶段结束后必须停止
   - 输出“本阶段完成内容 / 修改文件列表 / 如何验证 / 已知风险 / 下一阶段计划”
   - **在我没有明确回复“继续”之前，不要进入下一阶段**

3. **不要一次性输出整个项目所有代码**
   - 先规划
   - 再搭框架
   - 再实现最小可运行版本
   - 再补强健壮性
   - 再补 README 和对接接口

4. **优先保证可运行性，不优先追求炫技**
   - 先跑通最小链路
   - 后续再优化
   - 任何复杂增强都必须建立在已有脚本能运行的前提上

5. **必须严格遵守边界**
   - 你只实现“我的部分”
   - 不要越界去实现合作者负责的大段 ProPainter 主链、全量 evaluation、最终论文/报告自动化、Part 3 全部扩展

---

## 1. 项目背景与我负责的范围

这是一个 **Video Object Removal & Inpainting** 课程项目。项目要求的核心是：  
从含动态目标的视频中，自动识别并移除动态目标，并利用时间信息恢复背景。项目明确包含：
- Part 1：Hand-crafted baseline
- Part 2：SOTA mask extraction + video inpainting
- Part 3：至少一个优化/扩展方向
- mandatory datasets 上的视频结果、README、GitHub、可视化等都要交

### 但注意：我不是负责整个项目
我现在只负责下面这些模块：

#### A. 数据标准化与输入脚手架
把原视频统一成可复用、可对齐、可供后续模块消费的数据格式。

#### B. SAM2 动态目标 mask 提取主线
把原始视频变成逐帧二值 mask、overlay 可视化视频，以及标准化 metadata。

#### C. Hand-crafted baseline
实现作业要求的经典 baseline：
- 检测/分割动态类
- 稀疏光流判断动态
- `cv2.inpaint`
- 可选：轻量 temporal background borrowing（只做最小可运行版即可）

#### D. 输出接口
把我的输出整理成 teammate 可以直接接后续 inpainting / evaluation 的格式。

---

## 2. 明确的非目标（非常重要）

以下内容**不要在前几轮主动实现**，除非我明确说“继续做后半段”：

1. **不要主做 ProPainter 全链路**
   - 可以为它保留接口
   - 可以在 README 里说明如何对接
   - 但不要把它作为本次主交付的中心

2. **不要主做完整 evaluation 框架**
   - 只需要预留接口，或写轻量 shape / path consistency check
   - 不需要完整跑通所有 JM/JR / PSNR / SSIM 大实验

3. **不要主做 Part 3 扩展**
   - 不做 diffusion keyframe inpainting
   - 不做 SAM3 主线
   - 不做复杂多模型集成

4. **不要擅自重写整个 repo**
   - 只搭建这一部分所需的工程骨架
   - 保证与后续模块容易集成即可

---

## 3. 你要完成的最终交付物

你的最终交付物只包括以下内容：

### 3.1 数据脚手架
至少包含：
- `scripts/extract_frames.py`
- `scripts/compose_video.py`
- `scripts/check_alignment.py`
- `scripts/prepare_video.py`（可选：统一分辨率 / fps / pad / resize）
- `src/io/` 下的通用 I/O 工具

### 3.2 SAM2 pipeline
至少包含：
- `scripts/run_sam2_masks.py`
- `src/segmentation/sam2_pipeline.py`
- `src/mask_refine/postprocess.py`
- `src/visualization/overlay.py`

### 3.3 Hand-crafted baseline
至少包含：
- `scripts/run_handcrafted_baseline.py`
- `src/detection/detector.py`
- `src/motion/lk_motion_filter.py`
- `src/inpainting/baseline_inpaint.py`

### 3.4 配置与说明
至少包含：
- `configs/sam2.yaml`
- `configs/handcrafted.yaml`
- `README_my_part.md`

### 3.5 标准输出目录
至少支持以下结构：

```text
data/
  <video_name>/
    raw/
      video.mp4
    processed/
      frames/
        00000.png
        00001.png
      masks_sam2/
        00000.png
        00001.png
      masks_baseline/
        00000.png
        00001.png
      overlays/
        sam2_overlay.mp4
        baseline_overlay.mp4
      results_baseline/
        00000.png
        00001.png
      videos/
        baseline_result.mp4
      meta.json
```

---

## 4. 输出接口契约（必须严格遵守）

为了让我的 teammate 直接接后续模块，所有脚本必须遵守下面的 I/O 契约。

### 4.1 帧命名
统一 5 位零填充：
- `00000.png`
- `00001.png`
- ...

### 4.2 mask 格式
- 单通道 8-bit PNG
- 背景 = 0
- 前景/待移除区域 = 255

### 4.3 分辨率契约
- 同一个视频下：
  - `frames/*.png`
  - `masks_sam2/*.png`
  - `masks_baseline/*.png`
  - `results_baseline/*.png`
  必须完全同尺寸

### 4.4 meta.json 最少字段
```json
{
  "video_name": "tennis",
  "fps": 25,
  "width": 854,
  "height": 480,
  "num_frames": 120,
  "source_video": "data/tennis/raw/video.mp4",
  "frame_dir": "data/tennis/processed/frames",
  "sam2_mask_dir": "data/tennis/processed/masks_sam2",
  "baseline_mask_dir": "data/tennis/processed/masks_baseline"
}
```

### 4.5 CLI 约定
所有主脚本都必须支持：
- `--input_video`
- `--output_dir`
- `--config`
- `--device`
- `--save_overlay`

---

## 5. 技术选型要求

### 5.1 SAM2 主线
SAM2 是 mask extraction 的主线。  
要求：
- 优先使用官方/标准推理接口
- 从视频得到逐帧 mask
- 若需要 prompt，优先支持：
  - bbox prompt
  - point prompt
  - optional manual init prompt json

### 5.2 Hand-crafted baseline
优先组合：
- `YOLOv8-seg` 或 `Mask R-CNN` 做候选动态类 mask
- `Lucas-Kanade sparse optical flow` 做动态判断
- `cv2.inpaint` 做基础 inpainting
- 轻量 mask dilation / morphology 后处理

### 5.3 可选增强
只在最小链路跑通后才考虑：
- 全局运动补偿（可选）
- temporal background borrowing（简化版）
- 更强的 mask smooth

---

## 6. 代码风格与工程要求

### 6.1 工程质量
必须做到：
- Python 3.10+
- 模块化
- 函数有类型注解
- 有 docstring
- 有基础错误处理
- 路径用 `pathlib`
- 配置从 yaml 读入
- 不要把所有逻辑塞在一个脚本里

### 6.2 日志
统一使用 `logging`，不要满屏 `print`
至少输出：
- 输入路径
- 输出路径
- 分辨率
- 帧数
- 每个阶段完成状态
- 失败原因

### 6.3 可测试性
至少提供最小冒烟验证：
- 一段短视频能抽帧
- mask 数量和 frame 数量一致
- overlay 视频能导出
- baseline 输出图像数量正确

---

## 7. Sub-agent 协作协议（Codex 必须内部遵守）

虽然你最终只返回一个整合答案，但你**必须先在内部按 Sub-agent 分工思考并执行**。

### 7.1 Orchestrator Agent
职责：
- 读取需求
- 划定边界
- 生成本轮实施计划
- 分配子任务给其他 agents
- 在本轮结束时汇总结果
- 明确停止点，等待我回复“继续”

### 7.2 Data/IO Agent
职责：
- 设计目录结构
- 实现视频抽帧/重组
- 检查 frame/mask/result 的分辨率一致性
- 输出 `meta.json`

### 7.3 SAM2 Agent
职责：
- 设计并实现 `run_sam2_masks.py`
- 管理 prompt 输入
- 导出逐帧 mask
- 做基础后处理和 overlay

### 7.4 Baseline Agent
职责：
- 候选动态类提取
- LK 光流动态判定
- dilation / morphology
- `cv2.inpaint` 最小链路

### 7.5 Validation Agent
职责：
- 写最小冒烟测试
- 加 shape assertions
- 检查路径、帧数、mask 数量一致性
- 失败时给出具体报错上下文

### 7.6 README/Integration Agent
职责：
- 写 `README_my_part.md`
- 明确输入输出契约
- 明确如何把 `masks_sam2/` 交给 teammate 的后续模块
- 说明哪些内容尚未实现

---

## 8. Prompt Chaining 规则（强制）

你必须严格按照下面的阶段工作。

# 阶段 0：只做规划，不写大段代码
你要做的事：
1. 理解项目边界
2. 输出 repo 结构建议
3. 输出本次只实现哪些模块
4. 输出阶段 1 的计划
5. **停止，等待我的“继续”**

禁止：
- 不要一次性生成全量代码
- 不要开始写 SAM2 / baseline 全实现

---

# 阶段 1：只搭工程骨架 + Data/IO
你要做的事：
1. 创建目录结构
2. 写公共 I/O 模块
3. 写抽帧、重组视频、对齐检查脚本
4. 写基础 config loader
5. 写最小 README 草稿
6. 给出如何运行的命令
7. **停止，等待我的“继续”**

阶段验收标准：
- 能对一个测试视频完成抽帧
- 能从抽出的帧重新合成视频
- 能检查尺寸/帧数一致性
- 能输出 `meta.json`

---

# 阶段 2：只实现 SAM2 pipeline
你要做的事：
1. 写 `run_sam2_masks.py`
2. 写 `sam2_pipeline.py`
3. 实现逐帧 mask 导出
4. 实现基础后处理
5. 实现 overlay 视频输出
6. **停止，等待我的“继续”**

阶段验收标准：
- 给定一个视频，可以产出：
  - `frames/`
  - `masks_sam2/`
  - `sam2_overlay.mp4`
- mask 和 frames 尺寸完全一致
- mask 数量 = frame 数量

---

# 阶段 3：只实现 hand-crafted baseline
你要做的事：
1. 写 detector 封装
2. 写 LK 光流动态判定
3. 写 mask 后处理
4. 写 `cv2.inpaint` baseline
5. 导出 baseline 视频和帧结果
6. **停止，等待我的“继续”**

阶段验收标准：
- 可以从原视频跑出 baseline mask
- 可以产出 `results_baseline/`
- 可以重组 `baseline_result.mp4`

---

# 阶段 4：只做稳定性与接口收尾
你要做的事：
1. 补参数检查
2. 补日志
3. 补 shape assertions
4. 补 README_my_part.md
5. 明确和 teammate 对接方式
6. **停止，等待我的“继续”**

阶段验收标准：
- 所有脚本可从 CLI 调用
- README 说明清楚输入输出
- teammate 可以直接消费你的 `frames + masks + meta`

---

## 9. 每一轮输出格式（Codex 必须照做）

每一轮回答必须严格按这个模板：

### A. 本轮目标
一句话说明本轮做什么。

### B. Sub-agent 分工摘要
简要列出本轮各 agent 做了什么。

### C. 修改/新增文件
逐个列出文件路径和作用。

### D. 核心实现
给出本轮代码。

### E. 如何运行
给出最少命令。

### F. 如何验证
给出检查项：
- 文件是否生成
- 尺寸是否一致
- 帧数是否一致
- 是否能成功导出视频

### G. 已知限制
说明当前未完成部分。

### H. 下一轮建议
说明下一轮该做什么。

### I. 停止
最后一行必须写：
**“本阶段到此为止。请回复‘继续’进入下一阶段。”**

---

## 10. 目录结构建议（目标形态）

```text
project_root/
├── README_my_part.md
├── requirements.txt
├── configs/
│   ├── sam2.yaml
│   └── handcrafted.yaml
├── scripts/
│   ├── prepare_video.py
│   ├── extract_frames.py
│   ├── compose_video.py
│   ├── check_alignment.py
│   ├── run_sam2_masks.py
│   └── run_handcrafted_baseline.py
├── src/
│   ├── common/
│   │   ├── config.py
│   │   ├── logging_utils.py
│   │   └── paths.py
│   ├── io/
│   │   ├── video_io.py
│   │   ├── frame_io.py
│   │   └── metadata.py
│   ├── segmentation/
│   │   └── sam2_pipeline.py
│   ├── detection/
│   │   └── detector.py
│   ├── motion/
│   │   └── lk_motion_filter.py
│   ├── mask_refine/
│   │   └── postprocess.py
│   ├── inpainting/
│   │   └── baseline_inpaint.py
│   ├── visualization/
│   │   └── overlay.py
│   └── validation/
│       └── sanity_checks.py
└── data/
```

---

## 11. 你对外部依赖的处理方式

### 11.1 依赖原则
- 能直接 `pip install` 的尽量直接写进 `requirements.txt`
- 对大型外部仓库（如 SAM2），优先：
  - 提供清晰的安装说明
  - 封装调用接口
  - 不要把所有外部代码直接复制到本项目里

### 11.2 若 SAM2 安装复杂
处理策略：
1. 先封装抽象接口
2. 允许在 config 中填写 checkpoint / config path
3. 如环境未就绪，报出明确错误，而不是静默失败

---

## 12. 针对“resolution 不一致”的专项要求

这是当前真实痛点，必须优先解决。

你必须保证：
1. 所有抽帧输出使用统一尺寸
2. 所有 mask 输出与对应 frame 完全同尺寸
3. baseline 输出帧与输入帧同尺寸
4. overlay 视频尺寸与 frame 一致
5. `check_alignment.py` 能自动报错以下问题：
   - frame 缺失
   - mask 缺失
   - 尺寸不匹配
   - 帧数不一致

---

## 13. 针对数据集的现实策略

优先支持这些输入：
- Wild video
- bmx-trees
- tennis

如果路径组织不同，不要硬编码；必须通过 CLI 和 config 控制。

暂时不要把 DAVIS 的复杂评测逻辑写死进来，但目录设计要能兼容未来接入。

---

## 14. 针对 teammate 协作的接口说明

你的输出要让 teammate 能直接用：

### 14.1 对 SAM2 输出
给后续模块：
- `frames/`
- `masks_sam2/`
- `meta.json`

### 14.2 对 baseline 输出
给报告和对比图用：
- `masks_baseline/`
- `results_baseline/`
- `baseline_result.mp4`

### 14.3 README 必须写清楚
- 哪些是你负责的
- 哪些不是你负责的
- 后续模块应该从哪里读取你的输出

---

## 15. 首轮启动 Prompt（直接执行）

下面这段是你现在就应该执行的启动指令。

---

你现在开始执行 **阶段 0：规划阶段**。

请严格遵守以下要求：

1. 你必须采用 **Sub-agent 协作模式**，至少包含：
   - Orchestrator Agent
   - Data/IO Agent
   - SAM2 Agent
   - Baseline Agent
   - Validation Agent
   - README/Integration Agent

2. 你必须使用 **Prompt Chaining**
   - 当前只做阶段 0
   - 不要进入阶段 1
   - 不要一次性写完整项目代码

3. 你必须严格限制范围
   - 只做我的部分：
     - 数据标准化与输入脚手架
     - SAM2 mask extraction pipeline
     - hand-crafted baseline
     - 对接输出接口
   - 不要做：
     - ProPainter 主链
     - 全量 evaluation
     - Part 3
     - 最终论文自动化

4. 你当前阶段只输出这些内容：
   - 任务边界重述
   - repo 目录结构建议
   - 模块划分
   - 每个 sub-agent 的本轮职责
   - 阶段 1 的实施计划
   - 风险点与依赖项

5. 你的回答最后必须写：
**“本阶段到此为止。请回复‘继续’进入下一阶段。”**

---

## 16. 第二轮启动 Prompt（只有我回复“继续”后才能执行）

如果我回复“继续”，你才可以执行下面这段：

---

现在进入 **阶段 1：工程骨架 + Data/IO**。

要求：
1. 先由 Orchestrator 总结本轮目标
2. Data/IO Agent 实现目录、抽帧、重组、alignment check、meta.json
3. Validation Agent 写最小 sanity checks
4. README/Integration Agent 写 README_my_part.md 初稿
5. 不要实现 SAM2 和 baseline 的核心逻辑
6. 每写完一批文件，都说明如何运行和验证
7. 回答结尾必须停止，并等待我再次回复“继续”

最后必须写：
**“本阶段到此为止。请回复‘继续’进入下一阶段。”**

---

## 17. 质量红线

以下问题绝对不允许出现：

- 一次性输出全项目所有代码
- 无视我的边界，主动去做合作者负责的后半段
- 没有停止，直接跨阶段继续
- 没有说明输入输出格式
- mask 和 frame 尺寸不一致
- 没有给出验证方法
- 大量硬编码路径
- 全逻辑堆在一个脚本里
- 对外部依赖失败时不给清晰错误信息

---

## 18. 成功标准

如果你做对了，这份“我的部分”应该达到：

1. 我能用一个视频跑通抽帧与重组
2. 我能用 SAM2 产出逐帧 mask 与 overlay
3. 我能用 hand-crafted baseline 产出基础结果视频
4. 我的 teammate 能直接读取我的输出目录
5. 我们后续能无痛把我的 `frames + masks + meta` 接到更强的 inpainting 模块

---

## 19. 最后一条提醒

你不是来“炫技”的。  
你是来帮我把**我负责的这一块**做成一个：
- 可运行
- 可交付
- 可和 teammate 对接
- 能支持后续实验与报告的工程模块

所以请始终优先：
**边界清晰 > 可运行 > 可验证 > 可对接 > 再谈优化**

