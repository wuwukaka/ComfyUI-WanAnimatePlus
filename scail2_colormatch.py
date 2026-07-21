import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np


def _warn(logger, message):
    if logger is not None:
        logger.warning(message)


def _info(logger, message):
    if logger is not None:
        logger.info(message)


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


def _shape_arg(shape):
    return ",".join(str(int(v)) for v in shape)


def _parse_shape_arg(value):
    return tuple(int(part) for part in value.split(",") if part)


def _decode_worker_metrics(stdout):
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def _format_mib(num_bytes):
    return f"{num_bytes / (1024 * 1024):.1f} MiB"


def _shared_memory_skip_reason(video_np, ref_np):
    if not sys.platform.startswith("linux"):
        return None

    shm_dir = "/dev/shm"
    required = int((video_np.nbytes * 2 + ref_np.nbytes) * 1.10)
    if not os.path.isdir(shm_dir):
        return (
            f"{shm_dir} is unavailable; required={_format_mib(required)}; "
            "skipping shared-memory worker"
        )
    if not os.access(shm_dir, os.R_OK | os.W_OK):
        return (
            f"{shm_dir} is not readable/writable; required={_format_mib(required)}; "
            "skipping shared-memory worker"
        )
    try:
        free = shutil.disk_usage(shm_dir).free
    except Exception as e:
        return (
            f"could not inspect {shm_dir}: {type(e).__name__}: {e}; "
            f"required={_format_mib(required)}; skipping shared-memory worker"
        )
    if free < required:
        return (
            f"{shm_dir} free={_format_mib(free)} required={_format_mib(required)}; "
            "skipping shared-memory worker"
        )
    return None


def _run_shared_memory_worker(video_np, ref_np, method, timeout):
    """Run color_matcher in an isolated worker using shared memory instead of .npy files."""
    try:
        from multiprocessing import shared_memory
    except Exception as e:
        return {"ok": False, "fallback": True, "reason": f"shared_memory unavailable: {e}"}

    src_shm = ref_shm = out_shm = None
    try:
        video_np = np.ascontiguousarray(video_np.astype(np.float32, copy=False))
        ref_np = np.ascontiguousarray(ref_np.astype(np.float32, copy=False))
        skip_reason = _shared_memory_skip_reason(video_np, ref_np)
        if skip_reason is not None:
            return {"ok": False, "fallback": True, "reason": skip_reason}

        src_shm = shared_memory.SharedMemory(create=True, size=video_np.nbytes)
        ref_shm = shared_memory.SharedMemory(create=True, size=ref_np.nbytes)
        out_shm = shared_memory.SharedMemory(create=True, size=video_np.nbytes)

        copy_start = time.perf_counter()
        src_arr = np.ndarray(video_np.shape, dtype=video_np.dtype, buffer=src_shm.buf)
        ref_arr = np.ndarray(ref_np.shape, dtype=ref_np.dtype, buffer=ref_shm.buf)
        src_arr[...] = video_np
        ref_arr[...] = ref_np
        copy_in_seconds = time.perf_counter() - copy_start

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker-shm",
            "--src-shm",
            src_shm.name,
            "--src-shape",
            _shape_arg(video_np.shape),
            "--src-dtype",
            str(video_np.dtype),
            "--ref-shm",
            ref_shm.name,
            "--ref-shape",
            _shape_arg(ref_np.shape),
            "--ref-dtype",
            str(ref_np.dtype),
            "--out-shm",
            out_shm.name,
            "--out-shape",
            _shape_arg(video_np.shape),
            "--out-dtype",
            str(video_np.dtype),
            "--method",
            method,
        ]

        worker_start = time.perf_counter()
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return {
                "ok": False,
                "fallback": False,
                "reason": f"TimeoutExpired: timeout={timeout}s",
                "copy_in_seconds": copy_in_seconds,
                "worker_seconds": time.perf_counter() - worker_start,
            }

        worker_seconds = time.perf_counter() - worker_start
        metrics = _decode_worker_metrics(result.stdout)
        if result.returncode != 0:
            return {
                "ok": False,
                "fallback": False,
                "reason": _failure_reason(result=result),
                "copy_in_seconds": copy_in_seconds,
                "worker_seconds": worker_seconds,
                "metrics": metrics,
            }

        copy_out_start = time.perf_counter()
        out_arr = np.ndarray(video_np.shape, dtype=video_np.dtype, buffer=out_shm.buf).copy()
        copy_out_seconds = time.perf_counter() - copy_out_start
        if out_arr.shape != video_np.shape or not np.isfinite(out_arr).all():
            return {
                "ok": False,
                "fallback": False,
                "reason": f"invalid shared-memory worker output shape={out_arr.shape}, expected={video_np.shape}, or non-finite output",
                "copy_in_seconds": copy_in_seconds,
                "worker_seconds": worker_seconds,
                "copy_out_seconds": copy_out_seconds,
                "metrics": metrics,
            }

        return {
            "ok": True,
            "output": np.ascontiguousarray(np.clip(out_arr.astype(np.float32, copy=False), 0.0, 1.0)),
            "copy_in_seconds": copy_in_seconds,
            "worker_seconds": worker_seconds,
            "copy_out_seconds": copy_out_seconds,
            "metrics": metrics,
        }
    except Exception as e:
        return {"ok": False, "fallback": True, "reason": f"{type(e).__name__}: {e}"}
    finally:
        for shm in (src_shm, ref_shm, out_shm):
            if shm is None:
                continue
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass


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
    timeout = max(60, int(video_np.shape[0]) * 10)

    shm_result = _run_shared_memory_worker(video_np, ref_np, method, timeout)
    if shm_result.get("ok"):
        metrics = shm_result.get("metrics", {})
        _info(
            logger,
            f"SCAIL-2 colormatch shared-memory worker completed ({_context(stage, method, chunk_index)}): "
            f"copy_in={shm_result.get('copy_in_seconds', 0.0):.2f}s, "
            f"worker={shm_result.get('worker_seconds', 0.0):.2f}s, "
            f"transfer={float(metrics.get('transfer_seconds', 0.0)):.2f}s, "
            f"copy_out={shm_result.get('copy_out_seconds', 0.0):.2f}s, "
            f"frames={int(metrics.get('frames', video_np.shape[0]))}"
        )
        return shm_result["output"]

    reason = shm_result.get("reason", "unknown shared-memory worker failure")
    _warn(
        logger,
        f"SCAIL-2 colormatch shared-memory worker failed or unavailable ({_context(stage, method, chunk_index)}): "
        f"{reason}; falling back to .npy worker"
    )

    tmp_dir = tempfile.mkdtemp(prefix="wananimateplus_scail2_colormatch_")
    try:
        src_path = os.path.join(tmp_dir, "src.npy")
        ref_path = os.path.join(tmp_dir, "ref.npy")
        out_path = os.path.join(tmp_dir, "out.npy")
        np.save(src_path, video_np)
        np.save(ref_path, ref_np)

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


def _require_worker_args(args, names):
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing worker argument(s): {', '.join(missing)}")


def _run_color_match_npy_worker(args):
    _require_worker_args(args, ["src", "ref", "out", "method"])
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


def _run_color_match_shm_worker(args):
    _require_worker_args(
        args,
        [
            "src_shm",
            "src_shape",
            "src_dtype",
            "ref_shm",
            "ref_shape",
            "ref_dtype",
            "out_shm",
            "out_shape",
            "out_dtype",
            "method",
        ],
    )
    from multiprocessing import shared_memory
    from color_matcher import ColorMatcher

    src_shm = ref_shm = out_shm = None
    total_start = time.perf_counter()
    transfer_seconds = 0.0
    try:
        src_shape = _parse_shape_arg(args.src_shape)
        ref_shape = _parse_shape_arg(args.ref_shape)
        out_shape = _parse_shape_arg(args.out_shape)
        src_dtype = np.dtype(args.src_dtype)
        ref_dtype = np.dtype(args.ref_dtype)
        out_dtype = np.dtype(args.out_dtype)

        src_shm = shared_memory.SharedMemory(name=args.src_shm)
        ref_shm = shared_memory.SharedMemory(name=args.ref_shm)
        out_shm = shared_memory.SharedMemory(name=args.out_shm)

        video = np.ndarray(src_shape, dtype=src_dtype, buffer=src_shm.buf)
        ref = np.ndarray(ref_shape, dtype=ref_dtype, buffer=ref_shm.buf)
        out_video = np.ndarray(out_shape, dtype=out_dtype, buffer=out_shm.buf)

        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"invalid src shape {video.shape}")
        if ref.ndim != 3 or ref.shape[-1] != 3:
            raise ValueError(f"invalid ref shape {ref.shape}")
        if out_video.shape != video.shape:
            raise ValueError(f"invalid output shape {out_video.shape}, expected {video.shape}")

        cm = ColorMatcher()
        for frame_index, frame in enumerate(video):
            transfer_start = time.perf_counter()
            out = np.asarray(cm.transfer(src=frame, ref=ref, method=args.method), dtype=np.float32)
            transfer_seconds += time.perf_counter() - transfer_start
            if out.shape != frame.shape:
                raise ValueError(f"invalid output shape {out.shape}, expected {frame.shape}, frame={frame_index}")
            if not np.isfinite(out).all():
                raise ValueError(f"non-finite output, frame={frame_index}")
            out_video[frame_index] = out

        print(
            json.dumps(
                {
                    "frames": int(video.shape[0]),
                    "transfer_seconds": transfer_seconds,
                    "total_seconds": time.perf_counter() - total_start,
                }
            ),
            flush=True,
        )
    finally:
        for shm in (src_shm, ref_shm, out_shm):
            if shm is not None:
                shm.close()


def _worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-shm", action="store_true")
    parser.add_argument("--src")
    parser.add_argument("--ref")
    parser.add_argument("--out")
    parser.add_argument("--src-shm")
    parser.add_argument("--src-shape")
    parser.add_argument("--src-dtype")
    parser.add_argument("--ref-shm")
    parser.add_argument("--ref-shape")
    parser.add_argument("--ref-dtype")
    parser.add_argument("--out-shm")
    parser.add_argument("--out-shape")
    parser.add_argument("--out-dtype")
    parser.add_argument("--method")
    args = parser.parse_args()

    if args.worker_shm:
        _run_color_match_shm_worker(args)
    elif args.worker:
        _run_color_match_npy_worker(args)
    else:
        raise ValueError("missing worker mode")


if __name__ == "__main__":
    _worker_main()
