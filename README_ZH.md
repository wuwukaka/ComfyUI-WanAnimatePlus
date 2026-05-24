# ComfyUI-WanAnimatePlus

[English](./README.md) | [中文](./README_ZH.md)

为 ComfyUI 的 WanAnimate 视频生成管线提供多参考图注入与无缝视频衔接能力。

## 项目简介

`ComfyUI-WanAnimatePlus` 在原版 WanVideoWrapper 的 WanAnimate 流程上新增了两大核心输入：

- **prefix_frames**：允许用户传入 1~5 张额外参考图，实现多参考图引导生成
- **transition_video**：允许用户传入上一段视频的最后 21 帧，实现无缝视频衔接

同时使用时，两者自动协调画布布局和帧偏移，互不干扰。

适用场景：

- 多镜头视频串联生成
- 视频续写 / 延长
- 需要多参考图控制的动作迁移流程

## 效果展示

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

## 安装方式

将本仓库放入 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wuwukaka/ComfyUI-WanAnimatePlus.git
```

安装完成后重启 ComfyUI。

> **重要**：要使用 `prefix_frames` 和 `transition_video`，**必须**全链路替换为 WanAnimatePlus 版本节点。在同一个工作流中混用 WanAnimatePlus 节点和原版 WanVideoWrapper 节点会导致输出异常。

## 快速开始

1. 启动 ComfyUI，确认 `WanAnimatePlus` 分类下能看到完整节点链路
2. **将整个工作流链路替换**为 WanAnimatePlus 版本：`ModelLoader`、`VAELoader`、`ContextOptions`、`AnimateEmbeds`、`Sampler`、`Decode` 及配套节点
3. **不要**在同一个工作流中混用原版 WanVideoWrapper 节点
4. 根据需要接入 `prefix_frames` 或 `transition_video` 输入
5. 示例工作流见 `example_workflows/` 目录

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
- `WanAnimatePlus Scheduler` / `WanAnimatePlus Schedulerv2`
- `WanAnimatePlus Decode` / `WanAnimatePlus Encode`
- `WanAnimatePlus LoraSelectMulti` / `WanAnimatePlus SetLoRAs`
- `WanAnimatePlus BlockSwap` / `WanAnimatePlus SetBlockSwap`
- `WanAnimatePlus TorchCompileSettings`
- `WanAnimatePlus Uni3C ControlnetLoader` / `WanAnimatePlus Uni3C Embeds`

### WanAnimatePlus AnimateEmbeds

核心节点，替代原版 `WanVideoAnimateEmbeds`。

**新增输入：**

| 输入 | 说明 |
|------|------|
| `prefix_frames` | 允许用户传入 1~5 张额外参考图，实现多参考图引导生成 |
| `transition_video` | 允许用户传入上一段视频的最后 21 帧，实现无缝视频衔接 |

其他输入与原版 WanVideoAnimateEmbeds 一致：`vae`、`width`、`height`、`num_frames`、`ref_images`、`pose_images`、`face_images`、`bg_images`、`mask`、`start_ref_image`、`clip_embeds` 等。

## 项目结构

```text
ComfyUI-WanAnimatePlus/
├─ wanvideo/                 # WanVideo 核心模型代码
├─ nodes.py                  # WanAnimatePlus embeds / encode / decode 核心节点
├─ nodes_sampler.py          # WanAnimatePlus sampler / scheduler 核心节点
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

推荐 3 张。第一张为主要参考（占据 5 帧），后两张各占据 4 帧。超过 5 张会被自动截断。如果输入不足 3 张，节点也会正常工作，但覆盖范围会相应减小。

### 4. transition_video 需要多少帧？

会自动裁剪到 21 帧（不足则用首帧补齐）。21 像素帧对应约 6 个 latent 帧的过渡空间。

## 致谢

本项目修改自 [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)，致敬原作者对 WanVideo 生态的巨大贡献。

## 联系方式

- Bilibili: [@wuwukasi](https://space.bilibili.com/670281046)
- 邮箱: wuwukawayi@gmail.com

## 赞助

如果觉得本项目对你有帮助，欢迎请我喝杯咖啡！

<p align="center"><img src="docs/images/image_003.png" alt="微信赞助码" width="400"/></p>

## 许可证

基于原项目使用 **Apache License, Version 2.0** 协议发布。修改文件头部均已标注原作者版权和修改说明。
