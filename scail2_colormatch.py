import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def _warn(logger, message):
    if logger is not None:
        logger.warning(message)


def _context(stage, method, chunk_index=None, frame_index=None):
    parts = [f"stage={stage}", f"method={method!r}"]
    if chunk_index is not None:
        parts.append(f"chunk={chunk_index}")
    if frame_index is not None:
        parts.append(f"frame={frame_index}")
    return ", ".join(parts)


def _validate_video(video, ref, method, stage, logger, chunk_index=None):
    if method == "disabled":
        return None
    if video is None:
        _warn(logger, f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index)}): missing video; keeping original frames")
        return None
    if ref is None:
        _warn(logger, f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index)}): missing reference; keeping original frames")
        return None

    video = np.asarray(video, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if video.ndim != 4 or video.shape[-1] < 3 or video.shape[0] == 0:
        _warn(
            logger,
            f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index)}): "
            f"invalid video shape {video.shape}; keeping original frames",
        )
        return None
    if ref.ndim != 3 or ref.shape[-1] < 3 or ref.size == 0:
        _warn(
            logger,
            f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index)}): "
            f"invalid reference shape {ref.shape}; keeping original frames",
        )
        return None
    video = video[..., :3]
    ref = ref[..., :3]
    if not np.isfinite(video).all():
        _warn(logger, f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index)}): non-finite video input; keeping original frames")
        return None
    if not np.isfinite(ref).all():
        _warn(logger, f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index)}): non-finite reference input; keeping original frames")
        return None
    return np.ascontiguousarray(np.clip(video, 0.0, 1.0)), np.ascontiguousarray(np.clip(ref, 0.0, 1.0))


def _run_worker(src_path, ref_path, out_path, method, timeout):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--src",
        src_path,
        "--ref",
        ref_path,
        "--out",
        out_path,
        "--method",
        method,
    ]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)


def _failure_reason(result=None, exc=None):
    if exc is not None:
        return f"{type(exc).__name__}: {exc}"
    stderr = (result.stderr or "").strip() if result is not None else ""
    stdout = (result.stdout or "").strip() if result is not None else ""
    detail = stderr or stdout or "no subprocess output"
    return f"returncode={result.returncode}, {detail}" if result is not None else detail


def _should_retry_per_frame(reason):
    if "No module named 'color_matcher'" in reason:
        return False
    return True


def safe_color_match_video_bhwc(video, ref, method, stage, logger=None, chunk_index=None):
    """Run color_matcher in a subprocess so native crashes cannot kill ComfyUI."""
    validated = _validate_video(video, ref, method, stage, logger, chunk_index)
    if validated is None:
        return video
    video_np, ref_np = validated
    tmp_dir = tempfile.mkdtemp(prefix="wananimateplus_scail2_colormatch_")
    try:
        src_path = os.path.join(tmp_dir, "src.npy")
        ref_path = os.path.join(tmp_dir, "ref.npy")
        out_path = os.path.join(tmp_dir, "out.npy")
        np.save(src_path, video_np)
        np.save(ref_path, ref_np)

        timeout = max(60, int(video_np.shape[0]) * 10)
        try:
            result = _run_worker(src_path, ref_path, out_path, method, timeout)
            if result.returncode == 0 and os.path.exists(out_path):
                out = np.load(out_path)
                if out.shape == video_np.shape and np.isfinite(out).all():
                    return np.ascontiguousarray(np.clip(out.astype(np.float32, copy=False), 0.0, 1.0))
                _warn(
                    logger,
                    f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index)}): "
                    f"invalid worker output shape={getattr(out, 'shape', None)}, expected={video_np.shape}, "
                    "or non-finite output; keeping original frames",
                )
                return video_np
            reason = _failure_reason(result=result)
            if not _should_retry_per_frame(reason):
                _warn(
                    logger,
                    f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index)}): "
                    f"{reason}; keeping original frames",
                )
                return video_np
            _warn(
                logger,
                f"SCAIL-2 colormatch chunk worker failed ({_context(stage, method, chunk_index)}): "
                f"{reason}; retrying per-frame",
            )
        except subprocess.TimeoutExpired as e:
            _warn(
                logger,
                f"SCAIL-2 colormatch chunk worker timed out ({_context(stage, method, chunk_index)}): "
                f"timeout={timeout}s; retrying per-frame",
            )
        except Exception as e:
            _warn(
                logger,
                f"SCAIL-2 colormatch chunk worker failed ({_context(stage, method, chunk_index)}): "
                f"{_failure_reason(exc=e)}; retrying per-frame",
            )

        matched = []
        frame_src_path = os.path.join(tmp_dir, "frame_src.npy")
        frame_out_path = os.path.join(tmp_dir, "frame_out.npy")
        for frame_index, frame in enumerate(video_np):
            if os.path.exists(frame_out_path):
                try:
                    os.remove(frame_out_path)
                except OSError:
                    pass
            np.save(frame_src_path, frame[None])
            try:
                result = _run_worker(frame_src_path, ref_path, frame_out_path, method, 30)
                if result.returncode == 0 and os.path.exists(frame_out_path):
                    out = np.load(frame_out_path)
                    if out.shape == (1,) + frame.shape and np.isfinite(out).all():
                        matched.append(np.clip(out[0].astype(np.float32, copy=False), 0.0, 1.0))
                        continue
                    _warn(
                        logger,
                        f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index, frame_index)}): "
                        f"invalid worker output shape={getattr(out, 'shape', None)}, expected={(1,) + frame.shape}, "
                        "or non-finite output; keeping original frame",
                    )
                else:
                    _warn(
                        logger,
                        f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index, frame_index)}): "
                        f"{_failure_reason(result=result)}; keeping original frame",
                    )
            except subprocess.TimeoutExpired:
                _warn(
                    logger,
                    f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index, frame_index)}): "
                    "timeout=30s; keeping original frame",
                )
            except Exception as e:
                _warn(
                    logger,
                    f"SCAIL-2 colormatch skipped ({_context(stage, method, chunk_index, frame_index)}): "
                    f"{_failure_reason(exc=e)}; keeping original frame",
                )
            matched.append(frame)
        return np.ascontiguousarray(np.stack(matched, axis=0).astype(np.float32, copy=False))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--src", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    from color_matcher import ColorMatcher

    video = np.load(args.src).astype(np.float32, copy=False)
    ref = np.load(args.ref).astype(np.float32, copy=False)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"invalid src shape {video.shape}")
    if ref.ndim != 3 or ref.shape[-1] != 3:
        raise ValueError(f"invalid ref shape {ref.shape}")

    cm = ColorMatcher()
    matched = []
    for frame in video:
        out = np.asarray(cm.transfer(src=frame, ref=ref, method=args.method), dtype=np.float32)
        if out.shape != frame.shape:
            raise ValueError(f"invalid output shape {out.shape}, expected {frame.shape}")
        if not np.isfinite(out).all():
            raise ValueError("non-finite output")
        matched.append(out)
    np.save(args.out, np.stack(matched, axis=0).astype(np.float32, copy=False))


if __name__ == "__main__":
    _worker_main()
