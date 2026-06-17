# SCAIL2 Loop-Aware Fast Path

## Summary
把 SCAIL2 内置循环做成独立模式：`frame_window_size < num_frames` 自动 loop，`frame_window_size == num_frames` 继续走非 loop / context mode。SCAIL2 loop 不能因为 `loop_args` 被关掉，fast path 也不能因为 loop 本身失效。

这份计划会原样写进 `SCAIL2_LOOP_FAST_PATH_PLAN.md`。

## Timing Contract
- `frame_window_size` 仍表示单个 chunk 的总可见窗口长度，不要改成“新帧数”。
- 所谓“canvas 扩展 5”，指的是每个非首 chunk 的窗口头部预留 5 帧接力锚点，不是把 `canvas_expansion_px` 再加 5。
- stride 固定为 `frame_window_size - 5`，例如 `77 -> 72`。
- 像素帧 overlap 是 5 帧，VAE latent overlap 不是 5，而是按当前 4x 时间下采样折算成 2 帧；freeze mask 要在 latent 空间保护这 2 帧。
- 每个 chunk 的条件切片都按 `[chunk_start : chunk_start + frame_window_size]` 取，前 5 帧保留为上一段尾帧锚点，后面的帧才是本 chunk 新生成内容。
- 最后一段不足整窗时，像当前 `_fit_sequence` 一样用最后一帧补齐，再在最终输出处裁掉多余部分。
- chunk seed 必须纯随机，每个 chunk 用独立 64-bit seed，记录到日志或 `scail2_chunk_seeds` 元数据里，便于复查。

## Implementation Changes
- `nodes.py`
  - 给 SCAIL2 embeds 新增 `frame_window_size`。
  - 新增 `scail2_looping`、`scail2_previous_frame_count=5`、`scail2_requested_frames` 这类元数据，供 sampler 明确分支。
  - `single_frame_prefix_encoding` 保持默认 on；多参考在 loop 模式下强制走单帧编码。
  - 当前 prefix / ref / bg / transition 的 mask 语义全部保留，不改黑白底规则，不改 `preserve_main_ref_background` / `prefix_alpha_crop` / `replacement_mode` 的行为。
  - 现有 `canvas_expansion_px` 只继续用于 prefix/transition，不拿来实现 loop overlap。
- `nodes_sampler.py`
  - 新增独立 `scail2_looping` 分支，不复用 WanAnimate 的 `looping` 分支。
  - SCAIL2 loop branch 不再被 `loop_args` 作为 fast path blocker。
  - `context_options` 只允许非 loop；`frame_window_size < num_frames` 时如果还连着 context，直接报明确错误。
  - 每个 chunk：
    - 用纯随机 seed 创建本 chunk 的 generator。
    - 取上一 chunk 解码后的尾 5 帧，重新编码后放进当前 chunk 头部。
    - 这 5 帧在模型输入里保留，但在解码输出里裁掉，不参与最终拼接。
    - pose / pose mask / ref mask / prefix mask 都按同一 chunk 时间轴切片，前 5 帧对应的条件必须保留，不能置零、不能跳过。
    - chunk 0 仍可用现有 transition / freeze / canvas 逻辑；后续 chunk 只额外叠加 previous-frame 锚点。
  - `predict_with_cfg` 传局部 `scail_data`，不要污染全局状态。
  - 每个 chunk 解码后，立刻删除本 chunk latent、局部 pose/mask、局部 freeze tensors、局部 scail_data，只保留 CPU 上的已解码结果和下一 chunk 需要的 5 帧锚点。
  - 最终输出顺序：先拼接各 chunk 的去重结果，再按现有 `canvas_expansion_px` 裁前导扩展，最后裁到请求的 `num_frames`。

## Test Plan
- 非 loop 回归：
  - `frame_window_size == num_frames`，行为与当前非 loop 路径一致。
  - `frame_window_size == num_frames` 且接 `context_options`，仍走现有 context mode。
- loop 基础：
  - `num_frames=145, frame_window_size=77`，stride 应是 `72`。
  - 第二个 chunk 的前 5 帧必须是前一段尾 5 帧，且这 5 帧的 pose/mask/ref mask 还在。
  - 每个 chunk seed 都不同，不是顺序递增。
- 条件与裁剪：
  - transition 只影响首 chunk。
  - prefix / ref / bg / mask 的语义不因 loop 改变。
  - 最终输出帧数严格等于请求值。
- fast path：
  - SCAIL2 loop + `clip_embeds` 仍可进入 fast path。
  - SCAIL2 loop 不应因为 `loop_args` 自动 fallback。
- 内存：
  - 多 chunk 生成时显存不应随 chunk 数持续增长。
  - 上一 chunk 解码后 latent 及时释放，相关局部 pose/mask 也一起释放。

## Assumptions
- 第一版固定 `previous_frame_count=5`，不额外暴露参数。
- SCAIL2 loop 第一版不和 `samples` / vid2vid latent 注入混用；如果接了，直接报明确错误，避免双重接力。
- `frame_window_size` 会像当前 `num_frames` 一样被规范到 `4n+1`。
- 纯随机 chunk seed 是有意设计，优先满足“每段不同”，而不是单 seed 完全复现。
