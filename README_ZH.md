# ComfyUI-WanAnimatePlus

[English](./README.md) | [中文](./README_ZH.md)

为 ComfyUI 的 WanAnimate 视频生成管线提供多参考图注入与无缝视频衔接能力。

## 项目简介

`ComfyUI-WanAnimatePlus` 在 WanVideo 工作流上新增了五大功能：

### prefix_frames 与 transition_video

- **prefix_frames**：允许用户传入 1~5 张额外参考图，实现多参考图引导生成
- **transition_video**：允许用户传入上一段视频的最后 21 帧，实现无缝视频衔接

同时使用时，两者自动协调画布布局和帧偏移，互不干扰。

### Bernini

支持 Bernini 模型，允许用户传入源视频、参考视频或参考图像作为生成条件，支持 v2v、rv2v、r2v 和 t2v 任务。

适用场景：

- 多镜头视频串联生成
- 视频续写 / 延长
- 需要多参考图控制的动作迁移流程
- 视频编辑（源视频 + 参考图）
- 参考图生视频

### SCAIL-2 Embeds

新增基于 WanAnimatePlus 采样器适配的 `WanAnimatePlus SCAIL_2 Embeds` 节点，用于准备 SCAIL-2 的参考图、pose、colored pose mask、reference mask、prefix/transition 硬冻结 latent，以及与 prefix 对齐的 colored mask。

### SCAIL-2 Flow 官方兼容节点

新增 `WanAnimatePlus SCAIL_2 Flow Embeds`、`WanAnimatePlus SCAIL_2 Flow Sampler` 和 `WanAnimatePlus VAE Decode`。这套节点使用官方 ComfyUI 的 `MODEL`、`VAE`、`CONDITIONING`、`LATENT` 和 `IMAGE` 接口，方便接入官方流程，同时保留 WanAnimatePlus SCAIL-2 的 prefix、transition、bg、mask、loop colormatch 和二阶段采样行为。

### Animate2 官方兼容 Embeds

新增 `WanAnimatePlus Animate2 Embeds`，用于官方 ComfyUI Animate2 模型。它把 Plus 风格的 `ref_image`、`prefix_frames`、`bg_image` 映射到官方 Animate2 的 `CONDITIONING`（`concat_latent_image`、`concat_mask`、`pose_video_latent`）和 `LATENT`。采样仍走 SCAIL-2 同一套 `WanAnimatePlus SCAIL_2 Flow Sampler`，没有单独的 Animate2 Sampler。

## 效果展示

### RunningHub 在线运行示例

RunningHub 是一个在线 ComfyUI 算力平台。如果你的本地配置无法运行 AI 模型，可以在 RunningHub 上免费运行下方工作流；点击链接注册可领取 1000 积分。

| Scail2 超长视频二阶段采样无劣化示例流 | 超强全能 WanAnimate Plus 版工作流 |
|---|---|
| [![Scail2 超长视频二阶段采样无劣化示例流](docs/images/image_004.png)](https://www.runninghub.ai/zh-cn/post/2073345711527444480/?inviteCode=rh-v1565) | [![超强全能 WanAnimate Plus 版工作流](docs/images/image_005.png)](https://www.runninghub.ai/zh-cn/post/2073345712362110976/?inviteCode=rh-v1565) |
| https://www.runninghub.ai/zh-cn/post/2073345711527444480/?inviteCode=rh-v1565 | https://www.runninghub.ai/zh-cn/post/2073345712362110976/?inviteCode=rh-v1565 |

### prefix_frames 与 transition_video 使用示例

![使用方法](docs/images/image_001.png)

### prefix_frames 效果演示

[](https://github.com/user-attachments/assets/6df01023-5daa-42ab-9817-27a3b49bd6af)

### transition_video 效果演示

[](https://github.com/user-attachments/assets/4c6d2d29-dc21-406c-8ae9-5201d4cc416b)

## 功能特性

### prefix_frames（多参考图注入）

允许用户传入 1~5 张额外参考图。内部通过扩展画布像素空间，将参考图按帧分布编码到生成视频前部，使控制信号（pose / face）自动完成帧偏移协调，从而实现多参考图引导生成。

- 支持 1~5 张参考图，超过 5 张自动截断
- 参考图自动缩放到目标分辨率
- pose / face / bg / mask 等控制信号的帧偏移自动对齐

### transition_video（无缝视频衔接）

允许用户传入上一段视频的最后 21 帧。内部将这段视频像素帧直接写入生成画布的前部位置，并通过采样+反向补齐控制信号偏移，使当前生成的视频与前置片段无缝衔接。

- 与 prefix 同时使用时自动协调画布布局，两者互不干扰

### Bernini

通过 VAE 编码将源视频、参考视频和/或参考图像作为生成条件注入。支持 v2v、rv2v、r2v 和 t2v 任务，任务类型根据接入的输入自动推断。

- 参考图像保持原始宽高比
- 兼容 context window

### SCAIL-2

通过 `WanAnimatePlus SCAIL_2 Embeds` 提供 SCAIL-2 ref / pose / mask 条件注入。

- 编码 `ref_image`、`pose_images`、`pose_image_mask`、`prefix_frames`、`prefix_mask`、`bg_image` 和 `reference_image_mask`
- SCAIL-2 输入会在编码前对齐到 32 像素倍数
- 支持 animation / replacement 两种模式
- 支持单帧 prefix 参考编码和可选的 `transition_video` 硬冻结条件
- 默认情况下，`prefix_frames` 会作为全分辨率 reference latents 编码，不扩展输出画布；关闭 `single_frame_prefix_encoding` 后才使用 legacy 37 前置像素帧 prefix 布局
- 单帧 prefix 模式下，`prefix_mask` 走与 `reference_image_mask` 相同的 reference-mask 路径
- 支持 context window；非首窗口可以看到 prepend 的 prefix/transition 上下文，但这些 prepend 预测不会进入 overlap 融合

### Animate2

通过 `WanAnimatePlus Animate2 Embeds` 提供官方 Animate2 条件注入，采样走 `WanAnimatePlus SCAIL_2 Flow Sampler`。

- 把 `ref_image`、`prefix_frames`、`bg_image` 映射到官方 `concat_latent_image` / `concat_mask` / `pose_video_latent`
- `prefix_frames` 是最多 5 张额外冻结身份 latent，采样后裁掉
- `bg_image` 填未知画布（替代中灰），仍然生成
- `frame_window_size` 小于 `num_frames` 时，内循环用 5 帧 concat/mask 交接
- 不包装官方 `WanAnimate2Cache` / `PoseBranchCache`；需要缓存时直接用官方节点

## 安装方式

将本仓库放入 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wuwukaka/ComfyUI-WanAnimatePlus.git
```

安装完成后重启 ComfyUI。

> **重要**：要使用 `prefix_frames`、`transition_video`、`Bernini` 或 `SCAIL_2 Embeds`，**必须**全链路替换为 WanAnimatePlus 版本节点。在同一个工作流中混用 WanAnimatePlus 节点和原版 WanVideoWrapper 节点会导致输出异常。

## 快速开始

1. 启动 ComfyUI，确认 `WanAnimatePlus` 分类下能看到完整节点链路
2. **将整个工作流链路替换**为 WanAnimatePlus 版本：`ModelLoader`、`VAELoader`、`ContextOptions`、`AnimateEmbeds`、`Sampler`、`Decode` 及配套节点
3. **不要**在同一个工作流中混用原版 WanVideoWrapper 节点
4. SCAIL-2 和 WanAnimate 工作流推荐优先使用 `WanAnimatePlus Easy Sampler` 或 `WanAnimatePlus Easy SamplerSettings`，它们只精简常用可见参数，底层仍保留完整采样功能，搭建更方便
5. 如果要接入官方 ComfyUI `MODEL/VAE/CONDITIONING/LATENT/IMAGE` 流程：
   - SCAIL-2：`WanAnimatePlus SCAIL_2 Flow Embeds` -> `WanAnimatePlus SCAIL_2 Flow Sampler` -> `WanAnimatePlus VAE Decode`
   - Animate2：`WanAnimatePlus Animate2 Embeds` -> `WanAnimatePlus SCAIL_2 Flow Sampler` -> `WanAnimatePlus VAE Decode`
6. 根据需要接入 `prefix_frames` 或 `transition_video` 输入
7. 示例工作流见 `example_workflows/` 目录

## 节点说明

WanAnimatePlus 暴露了一套完整工作流链路，用于避免与原版 WanVideoWrapper 节点跨包混用。

核心节点：

- `WanAnimatePlus ModelLoader`
- `WanAnimatePlus VAELoader`
- `WanAnimatePlus TextEncodeCached`
- `WanAnimatePlus ClipVisionEncode`
- `WanAnimatePlus ContextOptions`
- `WanAnimatePlus AnimateEmbeds`
- `WanAnimatePlus Sampler` / `WanAnimatePlus Samplerv2`
- `WanAnimatePlus Easy Sampler` / `WanAnimatePlus Easy SamplerSettings`
- `WanAnimatePlus Scheduler` / `WanAnimatePlus Schedulerv2`
- `WanAnimatePlus Decode` / `WanAnimatePlus Encode`
- `WanAnimatePlus LoraSelect` / `WanAnimatePlus LoraSelectMulti` / `WanAnimatePlus SetLoRAs`
- `WanAnimatePlus BlockSwap` / `WanAnimatePlus SetBlockSwap`
- `WanAnimatePlus TorchCompileSettings`
- `WanAnimatePlus SamplerExtraArgs`
- `WanAnimatePlus Uni3C ControlnetLoader` / `WanAnimatePlus Uni3C Embeds`
- `WanAnimatePlus Bernini`
- `WanAnimatePlus SCAIL_2 Embeds`
- `WanAnimatePlus SCAIL_2 Flow Embeds`
- `WanAnimatePlus SCAIL_2 Flow Sampler`
- `WanAnimatePlus Animate2 Embeds`
- `WanAnimatePlus VAE Decode`

### WanAnimatePlus Easy Sampler / Easy SamplerSettings

推荐 SCAIL-2 和 WanAnimate 工作流优先使用这两个节点。它们只把节点面板中的可见参数精简为常用项，例如 `steps`、`cfg`、`shift`、`seed`、`force_offload` 和 `scheduler`，底层仍然走完整的 WanAnimatePlus 采样参数和功能路径。

`WanAnimatePlus Easy Sampler` 可直接采样；`WanAnimatePlus Easy SamplerSettings` 输出 `SAMPLER_ARGS`，适合继续接入 `WanAnimatePlus SamplerFromSettings`。

### WanAnimatePlus AnimateEmbeds

核心节点，替代原版 `WanVideoAnimateEmbeds`。

**新增输入：**

| 输入 | 说明 |
|------|------|
| `prefix_frames` | 允许用户传入 1~5 张额外参考图，实现多参考图引导生成 |
| `transition_video` | 允许用户传入上一段视频的最后 21 帧，实现无缝视频衔接 |

其他输入与原版 WanVideoAnimateEmbeds 一致：`vae`、`width`、`height`、`num_frames`、`ref_images`、`pose_images`、`face_images`、`bg_images`、`mask`、`start_ref_image`、`clip_embeds` 等。

### WanAnimatePlus Bernini

为 Bernini 模型生成条件 latents，支持源视频、参考视频和参考图像。

**输入：**

| 输入 | 说明 |
|------|------|
| `vae` | 用于编码的 VAE 模型 |
| `width` / `height` / `num_frames` | 输出尺寸 |
| `source_video` | 待编辑/重建的源视频（v2v/rv2v），缩放到 width/height |
| `reference_video` | 要合成到源视频中的动态内容（视频插入），保持原始宽高比 |
| `reference_images` | 作为上下文 token 注入的参考图像（r2v/rv2v），保持原始宽高比 |
| `ref_max_size` | 参考媒体长边最大尺寸（默认 848） |
| `force_offload` | 编码后将 VAE 卸载以节省显存 |
| `tiled_vae` | 使用分块 VAE 编码以节省显存 |

任务类型（v2v、rv2v、r2v、t2v）根据连接的输入自动推断。

### WanAnimatePlus SCAIL_2 Embeds

为 WanAnimatePlus 采样生成 SCAIL-2 条件。请配合包含 pose / mask stream 的 SCAIL-2 checkpoint 使用。

**输入：**

| 输入 | 说明 |
|------|------|
| `vae` | 用于编码的 VAE |
| `width` / `height` / `num_frames` | 目标尺寸；宽高会对齐到 32 像素倍数 |
| `ref_image` | SCAIL-2 参考图条件 |
| `bg_image` | animation 模式下可选的单张背景图；单帧模式下会额外编码为 background reference latent，legacy canvas-prefix 模式下追加到用户 `prefix_frames` 之后；replacement 模式下忽略 |
| `pose_images` | 驱动 pose 视频/图像，按半分辨率编码 |
| `pose_image_mask` | colored per-identity pose mask 序列 |
| `prefix_mask` | 可选，与 `prefix_frames` 对齐的 colored mask 图像；在单帧 prefix 模式下走 reference-mask 路径，在 legacy canvas-prefix 模式下按 `1+4+4...` 展开并写入 prefix 像素 mask 帧 |
| `reference_image_mask` | colored reference mask 图像 |
| `replacement_mode` | 启用 SCAIL-2 replacement-mode RoPE 和 reference-mask 合成 |
| `preserve_main_ref_background` | 仅 animation 模式使用；开启时保留主参考图背景，关闭时使用 `reference_image_mask` 做黑底 alpha crop。replacement 模式下忽略 |
| `single_frame_prefix_encoding` | 将 `prefix_frames` 编码为独立的全分辨率 reference latents，而不是扩展画布；默认开启 |
| `prefix_frames` | 可选 prefix 图像。默认单帧模式下会成为 reference-stream latents；关闭单帧模式后才硬冻结到前置画布 |
| `transition_video` | 可选，硬冻结在 latent 序列前部的 transition 帧 |
| `clip_embeds` | 可选，来自 `WanAnimatePlus ClipVisionEncode` 的 CLIP vision 特征 |
| `force_offload` / `tiled_vae` | VAE 编码相关显存控制 |

`transition_colormatch` 可选择 `auto_drift`，用于 SCAIL-2 loop 模式的轻量段间色漂校正。它不调用完整颜色分布匹配算法，而是用上一段尾部最多 5 帧和当前段开头最多 5 帧的 RGB 均值检测跳变，并对当前输出段做轻量校正；连接 `transition_video` 时，第一段会使用 transition 尾部最多 5 帧作为参考。

短视频生成时 context window 可选；长视频或低显存场景建议使用 context window。context-window 模式下，单帧 prefix reference 会通过 SCAIL-2 reference stream 保持可见；legacy canvas prefix 和 transition latents 会 prepend 给模型作为上下文，并在 overlap 融合前移除这些 prepend 预测。

SCAIL-2 默认的 `single_frame_prefix_encoding` 模式不会因为 `prefix_frames` 扩展或裁剪输出。如果连接 `transition_video`，前置画布会扩展 21 个像素帧，解码后裁掉这 21 帧。关闭 `single_frame_prefix_encoding` 后，`prefix_frames` 使用 legacy 37 前置像素帧画布；同时连接 `transition_video` 时，transition 帧放在第 17-36 帧。

animation 模式下连接 `bg_image` 时，节点会为该背景图内部添加白色 mask。用户提供的 `prefix_frames` 和 `prefix_mask` 各自最多保留四张，让背景图占用剩余的 reference/prefix 容量。replacement 模式下 `bg_image` 会被忽略。

### WanAnimatePlus SCAIL_2 Flow Embeds / Flow Sampler / VAE Decode

这套节点用于官方 ComfyUI 类型链路，不依赖旧的 `WANVID...` 输入输出类型。`Flow Embeds` 接收官方 `CONDITIONING`、`VAE`、图片和 mask 输入，输出更新后的 `CONDITIONING` 与 `LATENT`；`Flow Sampler` 使用官方 sampler 核心采样；`WanAnimatePlus VAE Decode` 可直接解码普通 latent，也能读取 Flow Sampler 在内循环模式下写入 `LATENT` 对象的已解码视频。

Flow 节点保留 SCAIL-2 的主要行为：`bg_image` 在 animation 模式下占用一个 prefix/reference 容量，最终 reference 顺序保持为 `prefix -> main ref -> bg`；`transition_video` 使用 21 前置像素帧硬冻结并在输出时裁掉；loop 模式每个 chunk 使用随机 chunk seed；`transition_colormatch` 和 `auto_drift` 复用 WanAnimatePlus SCAIL-2 的段间校正逻辑。

Flow Embeds 中 `single_frame_prefix_encoding` 固定开启且不暴露。二阶段采样参数直接在 `Flow Sampler` 中提供：`phase1_mask`、`phase2_mask` 和 `phase2_start_step`。这里的 mask 表示保护强度，`1=冻结/保护`，`0=自由 denoise`。

当官方模型上游已经带有 context handler 时，Flow Sampler 会关闭内部 loop，交给官方 context 路径处理；此时二阶段设置只对内部 loop handoff chunk 生效，不会强制接管官方 context 采样。

同一套 Flow Sampler 也能跑 `WanAnimatePlus Animate2 Embeds`。它根据 latent 上的 Animate2 runtime 识别路径，采样后裁掉身份帧，内循环用 5 像素帧交接。二阶段、colormatch、canvas expansion、freeze mask 仍留在节点上，但对 Animate2 不生效。

### WanAnimatePlus Animate2 Embeds

官方 ComfyUI 兼容的 Animate2 条件注入，配合 Animate2 checkpoint 使用。节点口只有 `positive`、`negative`、`latent`；loop / prefix / bg 状态藏在 latent 里。采样用 `WanAnimatePlus SCAIL_2 Flow Sampler`，解码用 `WanAnimatePlus VAE Decode`。需要缓存时继续用官方 `WanAnimate2Cache` / `PoseBranchCache`。

**输入：**

| 输入 | 说明 |
|------|------|
| `positive` / `negative` | 官方文本 `CONDITIONING` |
| `vae` | 用于编码 ref / prefix / bg 画布 / pose 的 VAE |
| `width` / `height` / `num_frames` | 目标尺寸和输出长度；宽高对齐到 32 像素倍数，`num_frames` 对齐到 `4n+1` |
| `frame_window_size` | 内循环窗口。小于 `num_frames` 时，Flow Sampler 按 5 帧交接做内循环 |
| `batch_size` | latent batch 大小 |
| `pose_strength` / `ref_strength` | 官方 extra-cond 强度，不会直接乘到 latent 上 |
| `clip_vision_output` | 可选，身份/参考路径的 CLIP vision |
| `clip_vision_output_pose` | 可选，pose 路径的 CLIP vision；缺省时回退到 `clip_vision_output` |
| `positive_pose` | 可选，pose 分支文本 `CONDITIONING`；缺省时回退到 `positive` |
| `ref_image` | 官方身份槽（第 0 帧），冻结（`concat_mask=0`），采样后裁掉 |
| `prefix_frames` | 最多 5 张额外冻结身份 latent，接在官方 ref 槽后面；输出时裁掉，成片长度仍是 `num_frames` |
| `bg_image` | 可选单张图，用来填未知画布（替代中灰）；仍然生成（`concat_mask=1`） |
| `pose_images` | 驱动 pose 视频/图像。pose latent 时间维会补齐到 `pose_T = gen_T - 1` |
| `tiled_vae` | 可用时使用分块 VAE 编码 |

Animate2 不用 SCAIL-2 的 `replacement_mode`、prefix mask、二阶段采样或 loop colormatch。身份帧靠 `concat_mask` 冻结；`bg_image` 只是画布填充，不是第二个冻结 ref 槽。

内循环下过长的 pose 序列可能落到临时磁盘缓存。如果官方模型已经带有 context handler，Flow Sampler 会关闭内部 loop，交给官方 context 路径。

## 项目结构

```text
ComfyUI-WanAnimatePlus/
├─ wanvideo/                 # WanVideo 核心模型代码
├─ nodes.py                  # WanAnimatePlus embeds / encode / decode 核心节点
├─ nodes_sampler.py          # WanAnimatePlus sampler / scheduler 核心节点
├─ nodes_animate2.py         # 官方兼容 Animate2 Embeds 节点
├─ animate2_flow.py          # Animate2 prefix / bg / 5 帧内循环 helper
├─ scail2_flow.py            # 官方兼容 SCAIL-2 Flow helper
├─ scail2_loop_inputs.py     # Flow / Animate2 内循环临时磁盘缓存
├─ nodes_model_loading.py    # WanAnimatePlus model / VAE / LoRA / block swap 节点
├─ context_windows/          # Context window 调度
├─ cache_methods/            # 缓存加速方法
├─ utils.py                  # 公共工具函数
├─ docs/
│  └─ images/                # 文档图片
├─ example_workflows/        # 示例工作流
├─ __init__.py               # 节点注册入口
├─ pyproject.toml
├─ requirements.txt
└─ LICENSE
```

## 常见问题（FAQ）

### 1. 安装后看不到节点

- 确认仓库路径在 `ComfyUI/custom_nodes/ComfyUI-WanAnimatePlus`
- 确认已同时安装原版 `ComfyUI-WanVideoWrapper`
- 重启 ComfyUI 后在节点列表中搜索 `WanAnimatePlus`

### 2. 与原版节点冲突？

不会。本插件所有节点名使用 `WanAnimatePlus` 前缀，与原版 `WanVideo` 前缀完全不冲突，两者可同时安装。

### 3. prefix_frames 输入几张图合适？

推荐 3 张。第一张使用 1 帧，后续每张各展开 4 帧，即 `1+4+4...`，最多 5 张、17 帧。超过 5 张会被自动截断。如果输入不足 3 张，节点也会正常工作，但覆盖范围会相应减小。

### 4. transition_video 需要多少帧？

会自动裁剪到 21 帧（不足则用首帧补齐）。21 像素帧对应约 6 个 latent 帧的过渡空间。

### 5. 官方 Animate2 怎么跑？

用 `WanAnimatePlus Animate2 Embeds` -> `WanAnimatePlus SCAIL_2 Flow Sampler` -> `WanAnimatePlus VAE Decode`。没有单独的 Animate2 Sampler。缓存继续用官方 `WanAnimate2Cache` / `PoseBranchCache`。

## 致谢

本项目修改自 [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)，致敬原作者对 WanVideo 生态的巨大贡献。

感谢 [checknickname/ComfyUI-Scail2-Sampler-Helper](https://github.com/checknickname/ComfyUI-Scail2-Sampler-Helper) 提供 SCAIL-2 two-phase sampling 思路，也感谢 [user2318/ComfyUI-CustomNodeKit](https://github.com/user2318/ComfyUI-CustomNodeKit/) 提供实现参考。详细归属说明见 [NOTICE](NOTICE)。

## 联系方式

- Bilibili: [@wuwukasi](https://space.bilibili.com/670281046)
- 邮箱: wuwukawayi@gmail.com

## 赞助

如果觉得本项目对你有帮助，欢迎赞助支持！有了您的支持，我才有动力继续改进和维护这个项目。

每一份支持都很重要，让我能投入更多时间进行开发和新增功能。感谢！

<p align="center">
  <img src="docs/images/image_003.png" alt="微信赞助码" width="400"/>
  <img src="docs/images/image_002.jpg" alt="支付宝赞助码" width="400"/>
</p>

## 许可证

本项目是基于 [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) 的独立维护 fork / 衍生项目，并按 **Apache License, Version 2.0** 协议发布。再次感谢 kijai 以及原项目贡献者的工作。

本项目中的 wuwukasi/wuwukaka 修改部分和新增源码表达 Copyright (c) 2026 wuwukasi/wuwukaka。归属范围、作者声明和修改说明要求见 [NOTICE](NOTICE)。
