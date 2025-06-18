import os
import re
import json
import random
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

OPENERS = [
    "The video is captured from a camera mounted on a car. The camera is facing forward."
]

WEATHER_DESCRIPTIONS = {
    "ClearNoon": [
        "It is noon on a bright, cloud‑free day.",
        "Sunlight is strong and shadows are crisp under a clear midday sky.",
        "The road is bathed in uncompromised daylight around lunchtime.",
        "A few distant treetops sway gently beneath the bluest of skies.",
        "There are no clouds to soften the stark midday contrast.",
        "Traffic signs gleam under the direct overhead sun."
    ],
    "WetNoon": [
        "The pavement is slick after recent rain, though skies are beginning to clear.",
        "Reflections shimmer on the wet road at noon.",
        "Small puddles dot the asphalt, mirroring passing vehicles.",
        "Water beads trickle off nearby guardrails under timid midday light.",
        "Sunbreaks pierce scattered clouds, causing sudden glare off the roadway.",
        "Tires hiss as they cut through thin surface water."
    ],
    "HardRainNoon": [
        "Heavy rain is falling at midday, streaking the windshield.",
        "Torrential noon rain limits visibility and drenches the roadway.",
        "Wipers sweep frantically against a curtain of water.",
        "Headlights of oncoming cars glow in the gray downpour.",
        "Large droplets explode into mist upon impact with the hood.",
        "Roadside gutters rush with fast‑moving runoff."
    ],
    "ClearSunset": [
        "The sun is low and golden as evening approaches.",
        "A clear sunset bathes the scene in warm orange light.",
        "Long shadows stretch across the lanes toward the horizon.",
        "Distant hills are outlined by a fiery afterglow.",
        "A faint contrail cuts the pastel sky overhead.",
        "Street signs take on a gentle amber hue."
    ],
    "ClearNight": [
        "It is nighttime with a clear, moon‑lit sky.",
        "Streetlights illuminate the road under a star‑filled night.",
        "Constellations sparkle above the quiet highway.",
        "The asphalt appears almost silver under the full moon.",
        "Neon storefronts flicker in the distance.",
        "Signals and taillights stand out sharply in the darkness."
    ],
    "HardRainNight": [
        "Sheets of rain are pouring down after dark.",
        "A nighttime downpour creates sparkling spray in the headlights.",
        "Water cascades off rooftops and floods the gutters.",
        "Visibility drops to mere meters amid driving rain.",
        "Every passing vehicle leaves a swirling wake of mist.",
        "Lightning occasionally silhouettes the wet landscape."
    ],
}

TAIL_LIGHT_DESCRIPTIONS = {
    "left": [
        "Only the left rear indicator is blinking.",
        "The driver has activated the left turn signal.",
        "A rhythmic amber flash appears on the vehicle’s left side.",
        "Left indicator pulses steadily, hinting at an impending lane change.",
        "A single left‑side blinker cuts through the dusk.",
        "Left turn light flickers against the reflective road paint."
    ],
    "right": [
        "Only the right rear indicator is blinking.",
        "The driver has engaged the right turn signal.",
        "Amber pulses brighten the vehicle’s right tail lamp.",
        "Right indicator flashes in measured cadence toward the shoulder.",
        "A lone right‑side blinker signals an upcoming merge.",
        "Right rear lamp beats like a metronome in the rain."
    ],
    "hazard": [
        "Both rear indicators flash simultaneously, showing hazard lights.",
        "The vehicle's hazard lights are blinking on both sides.",
        "Twin amber lights pulse in unison, warning other drivers.",
        "Synchronized flashes suggest a potential roadside stop.",
        "Dual indicators blink rapidly to indicate distress.",
        "Both tail lamps strobe together, signaling caution to following traffic."
    ]
}


def build_prompt(weather_key: str, tail_key: str,
                 max_sentences: int = 3) -> str:
    opener     = random.choice(OPENERS)
    weather    = random.choice(WEATHER_DESCRIPTIONS[weather_key])
    tail_light = random.choice(TAIL_LIGHT_DESCRIPTIONS[tail_key])
    sents = [opener, weather, tail_light][:max_sentences]
    return " ".join(sents)


def rle_encode(mask: np.ndarray):
    mask = np.array(mask, order="F")
    encoded = mask_util.encode(np.array(mask.reshape(-1, 1), order="F"))
    return {"data": encoded, "mask_shape": mask.shape}


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

        prompt = build_prompt(weathers, mode)
        
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
