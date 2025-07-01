"""
TODO
1. txt 삭제
2. waymo_transfer1 양식
3. *.0000.mp4 -> *_0000.mp4
"""


import os
from glob import glob
from tqdm import tqdm
import multiprocessing

import ffmpeg

FPS      = 24
CLIP_FR  = 1440
SRC_DIR  = 'dataset2'
DST_DIR  = 'clip'


def numeric_frame(p: str):
    return int(os.path.basename(p).split('_')[0])


def seconds_to_timecode(seconds: float):
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    minutes = (millis % 3_600_000) // 60_000
    secs = (millis % 60_000) / 1000
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def make_clips(scene_path: str):
    rgb_dir  = os.path.join(scene_path, "rgb")
    overlay_dir  = os.path.join(scene_path, "overlay")
    tail_dir   = os.path.join(scene_path, "tail")

    clip_dir = os.path.join(DST_DIR, "videos")
    hdmap_dir = os.path.join(DST_DIR, "hdmap")
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(hdmap_dir, exist_ok=True)
    tailclip_dir = os.path.join(DST_DIR, "tail")
    os.makedirs(tailclip_dir, exist_ok=True)

    rgb_ext = '*.jpg'
    overlay_ext = '*.png'
    tail_ext = '*.png'

    rgb_paths = glob(os.path.join(rgb_dir, rgb_ext))
    overlay_paths  = glob(os.path.join(overlay_dir, overlay_ext))
    tail_paths   = glob(os.path.join(tail_dir, tail_ext))
    rgb_map = {numeric_frame(p): p for p in rgb_paths}
    overlay_map  = {numeric_frame(p): p for p in overlay_paths}
    tail_map     = {numeric_frame(p): p for p in tail_paths}

    common_ids = sorted(set(rgb_map) & set(overlay_map) & set(tail_map))
    common_ids = common_ids[15:]

    rgb_frames      = [rgb_map[idx]     for idx in common_ids]
    overlay_frames  = [overlay_map[idx] for idx in common_ids]
    tail_frames     = [tail_map[idx]    for idx in common_ids]
    total_frames    = len(common_ids)

    scene_name = os.path.basename(scene_path)

    for clip_idx, start in enumerate(range(0, total_frames, CLIP_FR)):
        end = start + CLIP_FR
        if end > total_frames:
            break

        list_path = os.path.join(clip_dir, f'_list_{scene_name}_{clip_idx:04d}.txt')
        with open(list_path, 'w') as f:
            for fp in rgb_frames[start:end]:
                f.write(f"file '{os.path.abspath(fp)}'\n")

        out_mp4 = os.path.join(
            clip_dir, f"{os.path.basename(scene_path)}_{clip_idx:04d}.mp4"
        )

        (
            ffmpeg
            .input(os.path.abspath(list_path), f='concat', safe=0, r=FPS)
            .output(out_mp4, vcodec='libx264', crf=18, pix_fmt='yuv420p', r=FPS)
            .global_args('-loglevel', 'error', '-hide_banner', '-y')
            .run(quiet=True, overwrite_output=True)
        )

        overlay_list_path = os.path.join(hdmap_dir, f'_list_{scene_name}_{clip_idx:04d}.txt')
        with open(overlay_list_path, 'w') as f:
            for fp in overlay_frames[start:end]:
                f.write(f"file '{os.path.abspath(fp)}'\n")

        overlay_mp4 = os.path.join(
            hdmap_dir, f"{scene_name}_{clip_idx:04d}.mp4"
        )

        (
            ffmpeg
            .input(os.path.abspath(overlay_list_path), f='concat', safe=0, r=FPS)
            .output(overlay_mp4, vcodec='libx264', crf=18, pix_fmt='yuv420p', r=FPS)
            .global_args('-loglevel', 'error', '-hide_banner', '-y')
            .run(quiet=True, overwrite_output=True)
        )

        # ---------- Tail‑light video ----------
        tail_list_path = os.path.join(tailclip_dir, f'_list_{scene_name}_{clip_idx:04d}.txt')
        with open(tail_list_path, 'w') as f:
            for fp in tail_frames[start:end]:
                f.write(f"file '{os.path.abspath(fp)}'\n")

        tail_mp4 = os.path.join(
            tailclip_dir, f"{scene_name}_{clip_idx:04d}.mp4"
        )

        (
            ffmpeg
            .input(os.path.abspath(tail_list_path), f='concat', safe=0, r=FPS)
            .output(tail_mp4, vcodec='libx264', crf=18, pix_fmt='yuv420p', r=FPS)
            .global_args('-loglevel', 'error', '-hide_banner', '-y')
            .run(quiet=True, overwrite_output=True)
        )


def main():
    scenes = glob(os.path.join(SRC_DIR, '*'))
    with multiprocessing.Pool(processes=16) as pool:
        list(tqdm(pool.imap_unordered(make_clips, scenes), total=len(scenes)))


if __name__ == "__main__":
    main()

