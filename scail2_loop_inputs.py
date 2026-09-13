# Copyright (c) 2026 wuwukasi/wuwukaka.
# Temporary disk-backed loop input cache for long pose/video sequences.
# Licensed under the Apache License, Version 2.0
import ctypes
import os
import shutil
from datetime import datetime

import torch
import folder_paths
from comfy import model_management as mm
from comfy.utils import common_upscale


INPUT_CACHE_DIR_PREFIX = "scail2_loop_input_cache_"
ANIMATE2_CACHE_DIR_PREFIX = "animate2_loop_input_cache_"
INPUT_CACHE_MARKER = ".wananimateplus_scail2_loop_input_cache"
ANIMATE2_CACHE_MARKER = ".wananimateplus_animate2_loop_input_cache"
KNOWN_CACHE_PREFIXES = (INPUT_CACHE_DIR_PREFIX, ANIMATE2_CACHE_DIR_PREFIX)
KNOWN_CACHE_MARKERS = (INPUT_CACHE_MARKER, ANIMATE2_CACHE_MARKER)
MAX_MEMORY_USED_RATIO = 0.93
MIN_FREE_MEMORY_BYTES = int(1.8 * 1024 ** 3)

_offload_device = mm.unet_offload_device()


def _system_memory_info():
    try:
        import psutil

        mem = psutil.virtual_memory()
        return int(mem.available), int(mem.total)
    except Exception:
        pass

    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys), int(status.ullTotalPhys)

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(avail_pages * page_size), int(phys_pages * page_size)
    except Exception:
        return None, None


def tensor_nbytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def dtype_nbytes(dtype):
    return int(torch.empty((), dtype=dtype).element_size())


def memory_allows_allocation(additional_bytes, max_used_ratio=MAX_MEMORY_USED_RATIO, min_free_bytes=MIN_FREE_MEMORY_BYTES):
    additional_bytes = max(int(additional_bytes or 0), 0)
    available, total = _system_memory_info()
    if available is None or total is None or total <= 0:
        return True
    remaining = available - additional_bytes
    if remaining < min_free_bytes:
        return False
    used_after = total - remaining
    return (used_after / total) <= max_used_ratio


def is_loop_sequence(value):
    return (
        isinstance(value, dict)
        and value.get("type") == "scail2_loop_sequence"
        and value.get("storage") in ("disk", "mixed")
    )


def is_disk_sequence(value):
    return is_loop_sequence(value)


def estimate_resized_bhwc_bytes(images, width, height):
    if images is None:
        return 0
    return int(images.shape[0]) * int(height) * int(width) * 3 * dtype_nbytes(images.dtype)


def safe_frame_chunk_size(frame_count, in_height, in_width, out_width, out_height, channels):
    # Keep CUDA pooling/interpolate calls well below the int32 element limit.
    element_budget = 64 * 1024 * 1024
    per_frame = max(in_height * in_width * channels, out_height * out_width * channels, 1)
    return max(1, min(int(frame_count), element_budget // per_frame))


def _is_input_cache_dir_name(name):
    for prefix in KNOWN_CACHE_PREFIXES:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        parts = suffix.split("_")
        if (
            len(parts) == 4
            and len(parts[0]) == 8 and parts[0].isdigit()
            and len(parts[1]) == 6 and parts[1].isdigit()
            and len(parts[2]) == 6 and parts[2].isdigit()
            and parts[3].isdigit()
        ):
            return True
    return False


def _norm_path(path):
    if path is None:
        return None
    try:
        return os.path.normcase(os.path.abspath(path))
    except Exception:
        return path


def cleanup_stale_input_cache_dirs(logger=None, exclude_paths=None):
    output_dir = folder_paths.get_output_directory()
    exclude = {_norm_path(path) for path in (exclude_paths or []) if path is not None}
    try:
        entries = list(os.scandir(output_dir))
    except Exception as e:
        if logger is not None:
            logger.warning(f"loop: failed to scan temporary input cache folders in {output_dir}: {e}")
        return

    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            if _norm_path(entry.path) in exclude:
                continue
            has_marker = any(os.path.exists(os.path.join(entry.path, marker)) for marker in KNOWN_CACHE_MARKERS)
            if not (_is_input_cache_dir_name(entry.name) or has_marker):
                continue
            shutil.rmtree(entry.path)
            if logger is not None:
                logger.info(f"loop: removed stale temporary input cache folder {entry.path}")
        except Exception as e:
            if logger is not None:
                logger.warning(f"loop: failed to remove stale temporary input cache folder {entry.path}: {e}")


def create_input_cache_dir(prefix=INPUT_CACHE_DIR_PREFIX, marker=INPUT_CACHE_MARKER, note=None):
    path = os.path.join(
        folder_paths.get_output_directory(),
        f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}",
    )
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, marker), "w", encoding="utf-8") as marker_file:
        marker_file.write(note or "WanAnimatePlus loop temporary input cache\n")
    return path


def remove_input_cache_dir(path, logger=None):
    if path is None:
        return None
    try:
        shutil.rmtree(path)
        if logger is not None:
            logger.info(f"loop: removed temporary input cache folder {path}")
        return None
    except Exception as e:
        if logger is not None:
            logger.warning(f"loop: failed to remove temporary input cache folder {path}: {e}")
        return path


def build_resized_bhwc_sequence(
    images,
    width,
    height,
    mode,
    crop,
    cache_state,
    name,
    segment_frames,
    logger=None,
    cache_prefix=INPUT_CACHE_DIR_PREFIX,
    cache_marker=INPUT_CACHE_MARKER,
    cache_note=None,
):
    images = images[:, :, :, :3]
    if images.shape[0] <= 0:
        raise ValueError("loop input sequence contains no frames")
    chunk_size = max(
        1,
        min(
            int(segment_frames),
            safe_frame_chunk_size(
                images.shape[0],
                images.shape[1],
                images.shape[2],
                width,
                height,
                3,
            ),
        ),
    )
    segments = []
    last_frame = None
    for start in range(0, images.shape[0], chunk_size):
        chunk = images[start:start + chunk_size]
        expected_bytes = int(chunk.shape[0]) * int(height) * int(width) * 3 * dtype_nbytes(images.dtype)
        keep_in_memory = memory_allows_allocation(expected_bytes)
        if chunk.shape[1] == height and chunk.shape[2] == width:
            resized = chunk[:, :, :, :3]
        else:
            resized = common_upscale(chunk.movedim(-1, 1), width, height, mode, crop).movedim(1, -1)
        if keep_in_memory:
            stored = resized.detach().to(_offload_device)
            try:
                if stored.untyped_storage().data_ptr() == images.untyped_storage().data_ptr():
                    stored = stored.clone()
            except Exception:
                pass
            stored = stored.contiguous()
            segments.append({
                "start": int(start),
                "frames": int(stored.shape[0]),
                "tensor": stored,
                "nbytes": tensor_nbytes(stored),
            })
            last_frame = stored[-1:].detach().clone()
            del stored
        else:
            if cache_state.get("path") is None:
                if not cache_state.get("cleaned", False):
                    cleanup_stale_input_cache_dirs(logger)
                    cache_state["cleaned"] = True
                cache_state["path"] = create_input_cache_dir(
                    prefix=cache_prefix,
                    marker=cache_marker,
                    note=cache_note,
                )
            stored = resized.detach().cpu().contiguous()
            cache_path = os.path.join(cache_state["path"], f"{name}_{start:06d}.pt")
            torch.save(stored, cache_path)
            segments.append({
                "start": int(start),
                "frames": int(stored.shape[0]),
                "path": cache_path,
                "nbytes": tensor_nbytes(stored),
            })
            last_frame = stored[-1:].clone()
            del stored
        del resized
        del chunk

    return {
        "type": "scail2_loop_sequence",
        "storage": "mixed",
        "name": name,
        "shape": (int(images.shape[0]), int(height), int(width), 3),
        "segments": segments,
        "last_frame": last_frame,
    }


class LoopSequenceReader:
    def __init__(self, source, dim, storage_device, name):
        self.source = source
        self.dim = int(dim)
        self.storage_device = storage_device
        self.name = name
        self.offset = 0
        self.tensor = None
        self.last_frame = None
        self.sequence = is_loop_sequence(source)
        self.segments = []
        self.last_frame_path = None
        self.total_length = 0

        if source is None:
            return
        if self.sequence:
            self.segments = list(source.get("segments", []))
            shape = tuple(source.get("shape", ()))
            if len(shape) == 0 or self.dim >= len(shape) or int(shape[self.dim]) <= 0:
                raise ValueError(f"{name} sequence is empty")
            self.total_length = int(shape[self.dim])
            self.last_frame = source.get("last_frame", None)
            if isinstance(self.last_frame, torch.Tensor):
                self.last_frame = self.last_frame.to(storage_device)
            self.last_frame_path = source.get("last_frame_path", None)
            if self.last_frame is None and self.last_frame_path is None:
                raise ValueError(f"{name} sequence is missing last-frame metadata")
            return

        if not isinstance(source, torch.Tensor):
            raise TypeError(f"{name} sequence must be a Tensor or disk-backed metadata")
        if source.shape[self.dim] <= 0:
            raise ValueError(f"{name} sequence contains no frames")
        self.tensor = source.to(storage_device)
        self.total_length = int(self.tensor.shape[self.dim])
        self.last_frame = self.tensor.narrow(self.dim, self.total_length - 1, 1).detach().clone()

    def __bool__(self):
        return self.source is not None

    def _load_tensor(self, path):
        try:
            return torch.load(path, map_location=self.storage_device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=self.storage_device)

    def _get_last_frame(self):
        if self.sequence:
            if self.last_frame is None:
                self.last_frame = self._load_tensor(self.last_frame_path)
            return self.last_frame
        return self.last_frame

    def slice(self, start, length):
        if self.source is None:
            return None
        start = int(start)
        length = int(length)
        if length <= 0:
            last = self._get_last_frame()
            return last.narrow(self.dim, 0, 0).clone()
        if self.sequence:
            return self._slice_segments(start, length)
        return self._slice_tensor(start, length)

    def _slice_tensor(self, start, length):
        local_start = start - self.offset
        if local_start < 0:
            raise RuntimeError(
                f"{self.name} reader was trimmed past requested frame {start} "
                f"(offset={self.offset})"
            )
        take = max(min(length, self.tensor.shape[self.dim] - local_start), 0)
        if take > 0:
            out = self.tensor.narrow(self.dim, local_start, take)
        else:
            out = self._get_last_frame().narrow(self.dim, 0, 0)
        if out.shape[self.dim] < length:
            pad_shape = [1] * out.ndim
            pad_shape[self.dim] = length - out.shape[self.dim]
            last = self._get_last_frame().repeat(*pad_shape)
            out = torch.cat([out, last], dim=self.dim)
        return out

    def _segment_tensor(self, segment):
        tensor = segment.get("tensor", None)
        if tensor is not None:
            if tensor.device != self.storage_device:
                tensor = tensor.to(self.storage_device)
                segment["tensor"] = tensor
            return tensor
        path = segment.get("path", None)
        if path is None:
            raise RuntimeError(f"{self.name} input segment is neither loaded nor backed by disk")
        return self._load_tensor(path)

    def _slice_segments(self, start, length):
        end = start + length
        parts = []
        copied_until = start
        for segment in self.segments:
            seg_start = int(segment["start"])
            seg_end = seg_start + int(segment["frames"])
            if seg_end <= start:
                continue
            if seg_start >= end:
                break
            overlap_start = max(start, seg_start)
            overlap_end = min(end, seg_end)
            if overlap_end <= overlap_start:
                continue
            if overlap_start > copied_until:
                raise RuntimeError(f"{self.name} sequence has a gap before frame {overlap_start}")
            chunk = self._segment_tensor(segment)
            parts.append(chunk.narrow(self.dim, overlap_start - seg_start, overlap_end - overlap_start))
            copied_until = overlap_end
            if copied_until >= end:
                break

        if parts:
            out = torch.cat(parts, dim=self.dim) if len(parts) > 1 else parts[0].clone().contiguous()
        else:
            out = self._get_last_frame().narrow(self.dim, 0, 0).clone()
        if out.shape[self.dim] < length:
            pad_shape = [1] * out.ndim
            pad_shape[self.dim] = length - out.shape[self.dim]
            out = torch.cat([out, self._get_last_frame().repeat(*pad_shape)], dim=self.dim)
        return out

    def preload_available(self, logger=None, max_segments=None):
        if not self.sequence:
            return 0
        loaded = 0
        for segment in self.segments:
            if max_segments is not None and loaded >= int(max_segments):
                break
            if segment.get("tensor", None) is not None:
                continue
            path = segment.get("path", None)
            if path is None:
                continue
            nbytes = int(segment.get("nbytes", 0))
            if nbytes > 0 and not memory_allows_allocation(nbytes):
                break
            try:
                tensor = self._load_tensor(path).contiguous()
            except Exception as e:
                if logger is not None:
                    logger.warning(f"loop: failed to load {self.name} input segment {path}: {e}")
                break
            if nbytes <= 0 and not memory_allows_allocation(tensor_nbytes(tensor)):
                del tensor
                break
            segment["tensor"] = tensor
            try:
                os.remove(path)
                segment.pop("path", None)
            except FileNotFoundError:
                segment.pop("path", None)
            except Exception as e:
                if logger is not None:
                    logger.warning(f"loop: loaded {self.name} segment but failed to remove disk cache {path}: {e}")
            loaded += 1
        if loaded and logger is not None:
            logger.info(f"loop: loaded {loaded} queued {self.name} input segment(s) into memory")
        return loaded

    def trim_before(self, global_keep_start, logger=None):
        if self.source is None:
            return 0
        global_keep_start = int(global_keep_start)
        if self.sequence:
            kept = []
            trimmed = 0
            for segment in self.segments:
                seg_end = int(segment["start"]) + int(segment["frames"])
                if seg_end <= global_keep_start:
                    path = segment.get("path", None)
                    segment.pop("tensor", None)
                    if path is not None:
                        try:
                            os.remove(path)
                            segment.pop("path", None)
                        except FileNotFoundError:
                            segment.pop("path", None)
                        except Exception as e:
                            kept.append(segment)
                            if logger is not None:
                                logger.warning(f"loop: failed to remove consumed {self.name} input segment: {e}")
                            continue
                    trimmed += 1
                else:
                    kept.append(segment)
            self.segments = kept
            return trimmed

        local_keep = global_keep_start - self.offset
        if local_keep <= 0:
            return 0
        current_len = int(self.tensor.shape[self.dim])
        local_keep = min(local_keep, current_len)
        suffix_len = current_len - local_keep
        if suffix_len <= 0:
            self.tensor = self.tensor.narrow(self.dim, current_len, 0).clone()
            self.offset += local_keep
            return 1

        suffix = self.tensor.narrow(self.dim, local_keep, suffix_len)
        if not memory_allows_allocation(tensor_nbytes(suffix)):
            if logger is not None:
                logger.warning(
                    f"loop: skipped {self.name} input trim because cloning the remaining suffix "
                    "would exceed the memory limit"
                )
            return 0
        self.tensor = suffix.clone().contiguous()
        self.offset += local_keep
        return 1
