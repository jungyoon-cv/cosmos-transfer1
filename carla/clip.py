import os
import ffmpeg
from glob import glob
from tqdm import tqdm


FPS      = 24
CLIP_FR  = 121
SRC_DIR  = 'tail_dataset'
DST_DIR  = 'tail_clip'
SUBS = ['rgb', 'control']


def numeric_frame(p: str) -> int:
    return int(os.path.basename(p).split('_')[0])


def make_clips(scene_path: str, sub: str):
    rgb_dir  = os.path.join(scene_path, sub)
    clip_dir = os.path.join(DST_DIR, sub)
    os.makedirs(clip_dir, exist_ok=True)

    ext = '*.jpg' if sub == 'rgb' else '*.png'
    frames = sorted(glob(os.path.join(rgb_dir, ext)), key=numeric_frame)

    for clip_idx, start in enumerate(range(0, len(frames), CLIP_FR)):
        end = start + CLIP_FR
        if end > len(frames):
            break

        list_path = os.path.join(clip_dir, f'_list_{clip_idx:04d}.txt')
        with open(list_path, 'w') as f:
            for fp in frames[start:end]:
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

        os.remove(list_path)


def main():
    scenes = glob(os.path.join(SRC_DIR, '*'))[120:130]

    for scene in scenes:
        for sub in SUBS:
            make_clips(scene, sub)


if __name__ == "__main__":
    main()