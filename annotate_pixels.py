# python
import argparse
import os
import cv2
import pandas as pd

def get_1fps_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(0, total_frames, max(1, fps)))
    cap.release()
    return frame_indices, fps

def annotate_frame_cv(frame, frame_idx):
    """
    Show frame in an OpenCV window and collect clicks.
    Returns (annotations, stop_all) where annotations is list of (x_img, y_img, pitch_id_str).
    Controls:
      - left click: register a point (then terminal will ask for pitch id string)
      - n: finish this frame and continue
      - r: reset annotations for this frame
      - q: quit entire session
    """
    window_name = f"Frame {frame_idx}"
    raw_clicks = []
    annotations = []
    stop_all = False

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            raw_clicks.append((int(x), int(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        disp = frame.copy()
        for (x, y, pid) in annotations:
            cv2.circle(disp, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(disp, pid, (int(x) + 6, int(y) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(50) & 0xFF

        # process any new clicks (prompt for a string pitch id)
        while raw_clicks:
            x, y = raw_clicks.pop(0)
            s = input(f"Enter pitch ID (string) for point ({x}, {y}) (or 'q' to quit, empty to skip): ").strip()
            if s.lower() in ("q", "quit"):
                stop_all = True
                break
            if s == "":
                print("Skipped point.")
                continue
            annotations.append((x, y, s))

        if stop_all:
            break

        if key == ord("n"):
            break
        if key == ord("r"):
            annotations.clear()
            print("Reset annotations for this frame.")
        if key == ord("q"):
            stop_all = True
            break

    cv2.destroyWindow(window_name)
    return annotations, stop_all

def main():
    parser = argparse.ArgumentParser(description="Annotate video frames (1 fps) with pixel->pitch string mappings using OpenCV windows.")
    parser.add_argument("--video", "-v", required=False,
                        default="/Users/ciarangray/Desktop/FYP-resources/data/Veo highlights Straffan IFC/040 002429_-_Scores_For.mp4",
                        help="Path to video file")
    parser.add_argument("--output", "-o", default="annotations.csv", help="Output CSV file for annotations")
    args = parser.parse_args()

    video_path = args.video
    out_path = args.output

    if not os.path.exists(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    frame_indices, fps = get_1fps_frames(video_path)
    cap = cv2.VideoCapture(video_path)
    all_rows = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        print(f"\nAnnotating frame {frame_idx} ({frame_idx / fps:.1f}s). Left-click to add points.")
        annotations, stop = annotate_frame_cv(frame, frame_idx)

        for x_img, y_img, pitch_id in annotations:
            all_rows.append({
                "frame_idx": frame_idx,
                "time_sec": frame_idx / fps,
                "x_img": x_img,
                "y_img": y_img,
                "pitch_id": pitch_id  # stored as string
            })

        if stop:
            print("Stopping early by user request.")
            break

    cap.release()

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} annotation rows to {out_path}")
    else:
        print("No annotations collected; nothing saved.")

if __name__ == "__main__":
    main()
