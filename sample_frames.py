import cv2
import os
import csv
from pathlib import Path

# === CONFIGURATION ===
INPUT_DIR = "/Users/ciarangray/Desktop/FYP/Data/Ballyteague/Veo highlights Ballyteague League"    # folder containing the mp4 clips
OUTPUT_DIR = "/Users/ciarangray/Desktop/FYP/Data/Ballyteague/Frames"
CSV_PATH = "/Users/ciarangray/Desktop/FYP/Data/Ballyteague/frame_index.csv"      # list of frames extracted
FRAMES_PER_CLIP = 75             # how many frames to sample from each clip (set to None for fps sampling)
SAMPLE_FPS = 5                # alternatively, set to e.g. 3 to sample 3 frames/sec (takes priority if not None)
VIDEO_EXTS = ".mp4"

# ======================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["clip_name", "frame_index", "timestamp_sec", "saved_path"])

    for video_path in Path(INPUT_DIR).rglob("*"):
        if video_path.suffix.lower() not in VIDEO_EXTS:
            continue

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            print(f"Skipping {video_path}, cannot read FPS.")
            continue


        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        if SAMPLE_FPS:
            # Sample every N frames based on desired sampling fps
            step = int(round(fps / SAMPLE_FPS))
            frame_indices = list(range(0, total_frames, step))
        else:
            # Evenly space FRAMES_PER_CLIP frames across clip
            step = max(1, total_frames // FRAMES_PER_CLIP)
            frame_indices = list(range(0, total_frames, step))[:FRAMES_PER_CLIP]

        clip_name = video_path.stem
        out_dir = Path(OUTPUT_DIR) / clip_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            out_path = out_dir / f"{clip_name}_frame{i:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            timestamp = frame_idx / fps
            writer.writerow([clip_name, frame_idx, f"{timestamp:.2f}", str(out_path)])

        cap.release()

print(f"Frame extraction complete. Frames saved to: {OUTPUT_DIR}")
print(f"Frame index CSV: {CSV_PATH}")
