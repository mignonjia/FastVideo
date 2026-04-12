import json
import os
import sys
import numpy as np
import glob
import subprocess
import math
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from tqdm import tqdm
from typing import List, Tuple, Optional

# Configuration
DEFAULT_BASE_OUTPUT_DIR = os.environ.get(
    "PROCESS_BASE_OUTPUT_DIR",
    "traindata_0204_1600",
)
DEFAULT_WORKERS = int(os.environ.get("PROCESS_WORKERS", "8"))
DEFAULT_NUM_SHARDS = int(os.environ.get("PROCESS_SHARDS", "64"))
DEFAULT_INFLIGHT_MULT = int(os.environ.get("PROCESS_INFLIGHT_MULT", "2"))
DEFAULT_USE_FIND = os.environ.get("PROCESS_USE_FIND", "1") not in ("0", "false", "False")
DEFAULT_CACHE_EPISODES = os.environ.get("PROCESS_CACHE_EPISODES", "1") not in ("0", "false", "False")

DEFAULT_FFMPEG_THREADS = int(os.environ.get("PROCESS_FFMPEG_THREADS", "1"))
DEFAULT_FFMPEG_PRESET = os.environ.get("PROCESS_FFMPEG_PRESET", "ultrafast")
DEFAULT_FFMPEG_CRF = int(os.environ.get("PROCESS_FFMPEG_CRF", "28"))

DATA_DIR = '.'
NUM_WORKERS = DEFAULT_WORKERS  # Number of parallel workers
BASE_OUTPUT_DIR = DEFAULT_BASE_OUTPUT_DIR
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'videos')

# Slicing Configuration
FRAME_START = 11
FRAME_COUNT = 81
OUTPUT_WIDTH = 832
OUTPUT_HEIGHT = 480
# MOUSE_SHIFT is now calculated dynamically inside process_episode
ZERO_FIRST_FRAME = False # If True, first frame action is [0,0...] and others shift back
NUM_SHARDS = DEFAULT_NUM_SHARDS # Split output into N parts

# ffmpeg configuration
FFMPEG_THREADS = DEFAULT_FFMPEG_THREADS
FFMPEG_PRESET = DEFAULT_FFMPEG_PRESET
FFMPEG_CRF = DEFAULT_FFMPEG_CRF

# ======== Validation helpers (from validate.py) ========
def _blocks_for_scheme_4(T: int) -> Tuple[Optional[List[int]], str]:
    """
    Return block lengths list for scheme=4.
    require (T-1) % 4 == 0 and T >= 5
    T -> [1] + [4]*((T-1)//4)
    """
    if T < 5:
        return None, f"scheme=4 requires T>=5 (got {T})"
    if (T - 1) % 4 != 0:
        return None, f"scheme=4 requires (T-1)%4==0 (got T={T})"
    k = (T - 1) // 4
    return [1] + [4] * int(k), ""


def _block_constant_check(x: np.ndarray, blocks: List[int], atol: float = 0.0) -> Tuple[bool, str]:
    """
    Check: within each block, all frames are identical to the first frame of that block.
    x shape: [T, D]
    """
    T = int(x.shape[0])
    if sum(blocks) != T:
        return False, f"sum(blocks)={sum(blocks)} != T={T}"

    start = 0
    for bi, L in enumerate(blocks):
        if L <= 0:
            return False, f"invalid block length {L} at block_idx={bi}"
        if L > 1:
            ref = x[start]  # [D]
            seg = x[start + 1 : start + L]  # [(L-1), D]
            if atol <= 0.0:
                bad = np.any(seg != ref, axis=1)
            else:
                bad = np.any(np.abs(seg - ref) > atol, axis=1)
            if np.any(bad):
                off = int(np.nonzero(bad)[0][0]) + 1
                return False, f"block not constant at block_idx={bi}, offset={off}, block_len={L}"
        start += L

    return True, "ok"


def _validate_action_scheme4(keyboard_arr: np.ndarray, mouse_arr: np.ndarray, atol: float = 0.0) -> Tuple[bool, str]:
    """
    Validate that both keyboard and mouse arrays follow scheme=4:
    every 4 frames should be consistent within each block.
    """
    T_kb = keyboard_arr.shape[0]
    T_ms = mouse_arr.shape[0]
    
    if T_kb != T_ms:
        return False, f"keyboard frames {T_kb} != mouse frames {T_ms}"
    
    blocks, err = _blocks_for_scheme_4(T_kb)
    if blocks is None:
        return False, err
    
    ok_kb, reason_kb = _block_constant_check(keyboard_arr, blocks, atol)
    if not ok_kb:
        return False, f"keyboard: {reason_kb}"
    
    ok_ms, reason_ms = _block_constant_check(mouse_arr, blocks, atol)
    if not ok_ms:
        return False, f"mouse: {reason_ms}"
    
    return True, "ok"

# ========================================================

def _parse_args():
    parser = argparse.ArgumentParser(description="Process MatrixGame episodes into sliced videos and action arrays.")
    parser.add_argument("data_dir", nargs="?", default=".", help="Dataset root directory to search for episodes.")
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default=DEFAULT_BASE_OUTPUT_DIR,
        help="Output directory root. Can also set PROCESS_BASE_OUTPUT_DIR.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Max parallel workers (default via PROCESS_WORKERS).")
    parser.add_argument("--shards", type=int, default=DEFAULT_NUM_SHARDS, help="Split outputs into N shards (default via PROCESS_SHARDS).")
    parser.add_argument("--inflight-mult", type=int, default=DEFAULT_INFLIGHT_MULT, help="Keep at most workers * inflight_mult tasks in flight.")
    parser.add_argument("--use-find", action="store_true", default=DEFAULT_USE_FIND, help="Use `find` for faster episode discovery.")
    parser.add_argument("--no-find", action="store_false", dest="use_find", help="Disable `find` and use python glob.")
    parser.add_argument("--cache-episodes", action="store_true", default=DEFAULT_CACHE_EPISODES, help="Cache episode list to speed repeated runs.")
    parser.add_argument("--no-cache-episodes", action="store_false", dest="cache_episodes", help="Disable caching episode list.")
    parser.add_argument("--ffmpeg-threads", type=int, default=DEFAULT_FFMPEG_THREADS, help="Pass `-threads N` to ffmpeg.")
    parser.add_argument("--ffmpeg-preset", type=str, default=DEFAULT_FFMPEG_PRESET, help="x264 preset used by ffmpeg.")
    parser.add_argument("--ffmpeg-crf", type=int, default=DEFAULT_FFMPEG_CRF, help="x264 CRF used by ffmpeg (lower=better quality).")
    parser.add_argument("--no-validate-4consistent", action="store_true", default=False, help="Skip the 4-frame block consistency check on actions.")
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="Path to ffmpeg binary.")
    parser.add_argument("--output-width", type=int, default=OUTPUT_WIDTH, help="Output video width (default 832).")
    parser.add_argument("--output-height", type=int, default=OUTPUT_HEIGHT, help="Output video height (default 480).")
    return parser.parse_args()

def _discover_episode_dirs(data_dir: str, cache_path: str | None, use_find: bool) -> list[str]:
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            episode_dirs = [line.strip() for line in f if line.strip()]
        if episode_dirs:
            return episode_dirs

    if use_find:
        # `find` is typically much faster than python glob on large directory trees.
        cmd = ["find", data_dir, "-type", "f", "-name", "video.mp4"]
        try:
            out = subprocess.check_output(cmd, text=True)
            video_files = [line for line in out.splitlines() if line.strip()]
        except subprocess.CalledProcessError as e:
            print(f"[WARN] find failed, falling back to python glob: {e}")
            video_files = glob.glob(os.path.join(data_dir, "**", "video.mp4"), recursive=True)
    else:
        video_files = glob.glob(os.path.join(data_dir, "**", "video.mp4"), recursive=True)

    episode_dirs = sorted({os.path.dirname(f) for f in video_files})

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            for d in episode_dirs:
                f.write(d + "\n")

    return episode_dirs

# Action Mapping Configuration
KEY_TO_INDEX = {
    'W': 0,
    'S': 1,
    'A': 2,
    'D': 3
}

CAM_VALUE = 0.1
VIEW_ACTION_TO_MOUSE = {
    "up": [CAM_VALUE, 0.0],
    "down": [-CAM_VALUE, 0.0],
    "left": [0.0, -CAM_VALUE],
    "right": [0.0, CAM_VALUE],
    "up_right": [CAM_VALUE, CAM_VALUE],
    "up_left": [CAM_VALUE, -CAM_VALUE],
    "down_right": [-CAM_VALUE, CAM_VALUE],
    "down_left": [-CAM_VALUE, -CAM_VALUE],
}

def move_action_to_multihot(move_action: str) -> list:
    """
    Convert move_action string to a 6-dim multihot vector.
    Supports single keys ('W') and combinations ('WA', 'WD', etc.)
    """
    vector = [0, 0, 0, 0, 0, 0]
    if not move_action:
        return vector
        
    act_up = move_action.upper()
    # Check for each key's presence in the string
    for key, index in KEY_TO_INDEX.items():
        if key in act_up:
            vector[index] = 1
            
    return vector

def view_action_to_mouse(view_action: str) -> list:
    return VIEW_ACTION_TO_MOUSE.get(view_action.lower(), [0.0, 0.0])

def get_action_type(episode_dir: str, data_dir: str) -> str:
    """Return the action-type folder name (first component of path relative to data_dir)."""
    rel = os.path.relpath(episode_dir, data_dir)
    return rel.split(os.sep)[0]


def process_episode(episode_dir, episode_id):
    video_path = os.path.join(episode_dir, 'video.mp4')

    possible_json_rel_paths = [
        os.path.join('mg', 'mg_actions.json'),
        'actions.json'
    ]
    json_path = None
    for rel_path in possible_json_rel_paths:
        full_path = os.path.join(episode_dir, rel_path)
        if os.path.exists(full_path):
            json_path = full_path
            break
    if json_path is None or not os.path.exists(video_path):
        print(f"Skipping {episode_dir}: Missing json or video")
        return None

    action_type = get_action_type(episode_dir, DATA_DIR)
    action_video_dir = os.path.join(BASE_OUTPUT_DIR, action_type, 'videos')
    os.makedirs(action_video_dir, exist_ok=True)

    # 1. Process Actions
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sorted_keys = sorted(data.keys(), key=lambda x: int(x))
    
    keyboard_list = []
    mouse_list = []
    
    first_kb_idx = -1
    first_ms_idx = -1

    for i, key in enumerate(sorted_keys):
        move_action = data[key].get('move_action', '')
        multihot = move_action_to_multihot(move_action)
        keyboard_list.append(multihot)
        if first_kb_idx == -1 and any(v > 0 for v in multihot):
            first_kb_idx = i
        
        view_action = data[key].get('view_action', '')
        mouse_vector = view_action_to_mouse(view_action)
        mouse_list.append(mouse_vector)
        if first_ms_idx == -1 and any(abs(v) > 0 for v in mouse_vector):
            first_ms_idx = i
    
    # Fixed MOUSE_SHIFT: only 11 or 12
    # Rule:
    #   if mouse[0] != mouse[1] -> 12
    #   else -> 11
    if len(mouse_list) >= 2:
        m0 = np.asarray(mouse_list[0], dtype=np.float32)
        m1 = np.asarray(mouse_list[1], dtype=np.float32)
        dynamic_mouse_shift = 1 if (not np.allclose(m0, m1, atol=1e-8)) else 0
    else:
        dynamic_mouse_shift = 0  # fallback (or return None)

    # Standard slicing
    sliced_kb = keyboard_list[FRAME_START : FRAME_START + FRAME_COUNT]
    sliced_ms = mouse_list[FRAME_START + dynamic_mouse_shift : FRAME_START + dynamic_mouse_shift + FRAME_COUNT]
    
    keyboard_arr = np.array(sliced_kb, dtype=np.float32)
    mouse_arr = np.array(sliced_ms, dtype=np.float32)
    
    # Validate: every 4 frames should be consistent (scheme=4)
    if not SKIP_VALIDATE_4CONSISTENT:
        is_valid, validation_reason = _validate_action_scheme4(keyboard_arr, mouse_arr)
        if not is_valid:
            return {
                "skipped_validation": True,
                "reason": validation_reason,
                "episode_id": episode_id,
                "episode_dir": episode_dir
            }
    
    action_dict = {
        'keyboard': keyboard_arr,
        'mouse': mouse_arr
    }
    
    # 2. File Naming
    filename_prefix = f"{episode_id:06d}"
    output_video_rel = f"{filename_prefix}.mp4"
    output_action_rel = f"{filename_prefix}_action.npy"
    output_image_rel = f"{filename_prefix}.jpg"

    output_video_full = os.path.join(action_video_dir, output_video_rel)
    output_action_full = os.path.join(action_video_dir, output_action_rel)
    
    # 3. Video Slicing
    ffmpeg_bin = FFMPEG_BIN
    ffmpeg_video_cmd = [
        ffmpeg_bin, '-y', '-hide_banner', '-loglevel', 'error', '-nostdin',
        '-i', video_path,
        '-vf', f"select='between(n,{FRAME_START},{FRAME_START + FRAME_COUNT - 1})',scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}",
        '-vsync', 'vfr',
        '-an',
        '-threads', str(FFMPEG_THREADS),
        '-c:v', 'libx264',
        '-preset', str(FFMPEG_PRESET),
        '-crf', str(FFMPEG_CRF),
        output_video_full
    ]
    
    try:
        # Save Video
        subprocess.run(ffmpeg_video_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        # Save Actions
        np.save(output_action_full, action_dict)
        
        return {
            "path": output_video_rel,
            "action_path": output_action_rel,
            "image_path": output_image_rel,
            "episode_id": episode_id,
            "action_type": action_type,
            "video_path": video_path  # Keep original for later image extraction
        }
    except subprocess.CalledProcessError as e:
        err = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode() if e.stderr else "")
        print(f"Error processing {episode_dir}: {err}")
        return None

args = _parse_args()
DATA_DIR = args.data_dir
BASE_OUTPUT_DIR = args.base_output_dir
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
SKIP_VALIDATE_4CONSISTENT = args.no_validate_4consistent
FFMPEG_BIN = args.ffmpeg_bin
OUTPUT_WIDTH = args.output_width
OUTPUT_HEIGHT = args.output_height
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
NUM_WORKERS = args.workers
NUM_SHARDS = args.shards
FFMPEG_THREADS = args.ffmpeg_threads
FFMPEG_PRESET = args.ffmpeg_preset
FFMPEG_CRF = args.ffmpeg_crf

# Find all directories that contain a video.mp4 file (cache for speed)
cache_path = os.path.join(BASE_OUTPUT_DIR, "episode_dirs.txt") if args.cache_episodes else None
episode_dirs = _discover_episode_dirs(DATA_DIR, cache_path=cache_path, use_find=args.use_find)

results = []
# Statistics for validation failures
skipped_validation_count = 0
skipped_keyboard_count = 0
skipped_mouse_count = 0
skipped_other_count = 0

print(f"Found {len(episode_dirs)} episodes. Processing with {NUM_WORKERS} workers...")

# process episodes with bounded in-flight tasks (reduces overhead & FS thrash)
max_inflight = max(1, NUM_WORKERS * max(1, args.inflight_mult))
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    it = enumerate(episode_dirs)
    future_to_ep = {}

    # prime the queue
    for _ in range(min(max_inflight, len(episode_dirs))):
        try:
            i, ep_dir = next(it)
        except StopIteration:
            break
        future_to_ep[executor.submit(process_episode, ep_dir, i)] = (i, ep_dir)

    pbar = tqdm(total=len(episode_dirs), desc="Processing Episodes")
    while future_to_ep:
        done, _ = wait(future_to_ep, return_when=FIRST_COMPLETED)
        for future in done:
            i, ep_dir = future_to_ep.pop(future)
            try:
                result = future.result()
                if result:
                    if result.get("skipped_validation"):
                        # Count validation failures
                        skipped_validation_count += 1
                        reason = result.get("reason", "")
                        if reason.startswith("keyboard:"):
                            skipped_keyboard_count += 1
                        elif reason.startswith("mouse:"):
                            skipped_mouse_count += 1
                        else:
                            skipped_other_count += 1
                    else:
                        results.append(result)
            except Exception as e:
                tqdm.write(f"Error processing {ep_dir}: {e}")
            pbar.update(1)

            # enqueue next
            try:
                i2, ep_dir2 = next(it)
                future_to_ep[executor.submit(process_episode, ep_dir2, i2)] = (i2, ep_dir2)
            except StopIteration:
                pass
    pbar.close()

# Sort results by episode_id to maintain order
results.sort(key=lambda x: x['episode_id'])

# Group results by action type
from collections import defaultdict
results_by_type: dict = defaultdict(list)
for res in results:
    results_by_type[res['action_type']].append(res)

# Write sharded JSON and merge files per action type
total_shards_written = 0
for action_type, type_results in sorted(results_by_type.items()):
    type_output_dir = os.path.join(BASE_OUTPUT_DIR, action_type)
    os.makedirs(type_output_dir, exist_ok=True)

    shard_size = math.ceil(len(type_results) / NUM_SHARDS)

    for i in range(NUM_SHARDS):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, len(type_results))
        shard_results = type_results[start_idx:end_idx]

        if not shard_results and i > 0:
            continue

        s_v2c = []
        s_meta = []
        for res in shard_results:
            s_v2c.append({
                "path": res["path"],
                "cap": [""],
                "action_path": res["action_path"],
                "resolution": {"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT},
                "num_frames": FRAME_COUNT,
                "fps": 25,
                "duration": round(FRAME_COUNT / 25, 2)
            })
            s_meta.append({
                "video_path": f"videos/{res['path']}",
                "action_path": f"videos/{res['action_path']}",
                "num_frames": FRAME_COUNT,
                "width": OUTPUT_WIDTH,
                "height": OUTPUT_HEIGHT,
                "episode_id": res["episode_id"]
            })

        suffix = f"_{i}" if NUM_SHARDS > 1 else ""
        v2c_name = f"video2caption{suffix}.json"
        meta_name = f"metadata{suffix}.json"
        merge_name = f"merge{suffix}.txt"

        with open(os.path.join(type_output_dir, v2c_name), 'w') as f:
            json.dump(s_v2c, f, indent=4)
        with open(os.path.join(type_output_dir, meta_name), 'w') as f:
            json.dump(s_meta, f, indent=4)
        with open(os.path.join(type_output_dir, merge_name), 'w') as f:
            f.write(f"{type_output_dir}/videos,{type_output_dir}/{v2c_name}\n")

        total_shards_written += 1

print(f"\nProcessing complete.")
print(f"  Total episodes found:      {len(episode_dirs)}")
print(f"  Valid episodes processed:  {len(results)}")
print(f"  Skipped (validation fail): {skipped_validation_count}")
print(f"    - keyboard not 4-consistent: {skipped_keyboard_count}")
print(f"    - mouse not 4-consistent:    {skipped_mouse_count}")
print(f"    - other reasons:              {skipped_other_count}")
print(f"\nAction types found: {sorted(results_by_type.keys())}")
for action_type, type_results in sorted(results_by_type.items()):
    print(f"  {action_type}: {len(type_results)} episodes")
print(f"\nTotal shards written: {total_shards_written}")
print(f"Files generated under {BASE_OUTPUT_DIR}/<action_type>/: videos/, and sharded caption/meta/merge files.")
