import os
import re
import json
import pickle
from glob import glob
from tqdm import tqdm

import ffmpeg
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util

FPS      = 24
CLIP_FR  = 121
SRC_DIR  = 'tail_dataset'
DST_DIR  = 'tail_clip'


def rle_encode(mask: np.ndarray) -> dict:
    mask = np.array(mask, order="F")
    encoded = mask_util.encode(np.array(mask.reshape(-1, 1), order="F"))
    return {"data": encoded, "mask_shape": mask.shape}


def numeric_frame(p: str) -> int:
    return int(os.path.basename(p).split('_')[0])


def seconds_to_timecode(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    minutes = (millis % 3_600_000) // 60_000
    secs = (millis % 60_000) / 1000
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def make_clips(scene_path: str):
    rgb_dir  = os.path.join(scene_path, "rgb")
    
    clip_dir = os.path.join(DST_DIR, "rgb")
    seg_dir = os.path.join(DST_DIR, "seg")
    metas_dir = os.path.join(DST_DIR, "metas")
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(metas_dir, exist_ok=True)

    ext = '*.jpg'
    frames = sorted(glob(os.path.join(rgb_dir, ext)), key=numeric_frame)

    scene_name = os.path.basename(scene_path)
    tokens = scene_name.split("_")
    weathers, mode = tokens[1], tokens[-1]

    weather_parts = re.findall(r"[A-Z][a-z]*", weathers)
    weather_list = ", ".join(weather_parts)
    
    """
    The video is captured from a camera mounted on a car. The camera is facing forward. 
    날씨 이야기~
    후미등 상태 이야기~ 
    """
    prompt = f"{weather_list}, {mode.capitalize()}".strip(", ")

    for clip_idx, start in enumerate(range(0, len(frames), CLIP_FR)):
        end = start + CLIP_FR
        if end > len(frames):
            break

        list_path = os.path.join(clip_dir, f'_list_{clip_idx:04d}.txt')
        with open(list_path, 'w') as f:
            for fp in frames[start:end]:
                f.write(f"file '{os.path.abspath(fp)}'\n")

        out_mp4 = os.path.join(
            clip_dir, f"{os.path.basename(scene_path)}.{clip_idx:04d}.mp4"
        )

        (
            ffmpeg
            .input(os.path.abspath(list_path), f='concat', safe=0, r=FPS)
            .output(out_mp4, vcodec='libx264', crf=18, pix_fmt='yuv420p', r=FPS)
            .global_args('-loglevel', 'error', '-hide_banner', '-y')
            .run(quiet=True, overwrite_output=True)
        )

        os.remove(list_path)

        imgs = [np.array(Image.open(fp)) for fp in frames[start:end]]
        arr = np.stack(imgs)  # [T, H, W, 3]
        left = arr[:, :, :, 1] > 127
        right = arr[:, :, :, 2] > 127
        detections = [
            {"phrase": "tail_light_left", "segmentation_mask_rle": rle_encode(left)},
            {"phrase": "tail_light_right", "segmentation_mask_rle": rle_encode(right)},
        ]
        pkl_path = os.path.join(
            seg_dir, f"{os.path.basename(scene_path)}.{clip_idx:04d}.pickle"
        )
        with open(pkl_path, "wb") as pf:
            pickle.dump(detections, pf)

        clip_name = f"{scene_name}.{clip_idx:04d}"
        txt_path = os.path.join(metas_dir, clip_name + ".txt")
        json_path = os.path.join(metas_dir, clip_name + ".json")
        with open(txt_path, "w") as tf:
            tf.write(prompt)
        
        duration_sec = (end - start) / FPS
        meta = {
            "clip_id": clip_name + ".mp4",
            "video_id": scene_name + ".mp4",
            "url": "",
            "span_start": "00:00:00.000",
            "span_end": seconds_to_timecode(duration_sec),
            "caption": prompt,
        }
        with open(json_path, "w") as jf:
            json.dump(meta, jf)


def main():
    scenes = glob(os.path.join(SRC_DIR, '*'))[120:121]

    for scene in scenes:
        make_clips(scene)


if __name__ == "__main__":
    main()