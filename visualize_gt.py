#!/usr/bin/env python3
"""Ground-truth visualizer for Ego4D Short-Term Action Anticipation (STA)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from explorer.ego4d_sta import StaSample, build_sta_samples, remaining_ttc
from explorer.pyav_reader import PyAVVideoReader
from explorer.paths import resolve_data_root

DISPLAY_SCALE = 1
DEFAULT_MAX_DISPLAY_WIDTH = 1280
KEY_LEFT = {2424832, 65361, 81, 2, 63234}
KEY_RIGHT = {2555904, 65363, 83, 3, 63235}


def upscale(img: np.ndarray, scale: int = 1) -> np.ndarray:
    if scale == 1:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def fit_display(img: np.ndarray, scale: int, max_width: int | None) -> np.ndarray:
    out = upscale(img, scale=scale)
    if max_width and out.shape[1] > max_width:
        h, w = out.shape[:2]
        r = max_width / w
        out = cv2.resize(out, (max_width, int(h * r)), interpolation=cv2.INTER_AREA)
    return out


def draw_text_block(img, lines, origin=(10, 28), line_h=26):
    x, y = origin
    for i, (text, color) in enumerate(lines):
        if not text:
            continue
        yy = y + i * line_h
        cv2.putText(img, text, (x + 1, yy + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_timeline(img, sample: StaSample, current_frame: int):
    h, w = img.shape[:2]
    bar_h = 24
    y0 = h - bar_h - 10
    x0, x1 = 20, w - 20
    cv2.rectangle(img, (x0, y0), (x1, y0 + bar_h), (40, 40, 40), -1)

    span_start = sample.clip_start_frame
    span_end = max(sample.clip_end_frame, sample.annotation_frame + 1)
    span = max(1, span_end - span_start)

    def xpos(frame_idx: int) -> int:
        rel = (frame_idx - span_start) / span
        return int(x0 + rel * (x1 - x0))

    if sample.action_start_frame is not None and sample.action_end_frame is not None:
        cv2.rectangle(
            img,
            (xpos(sample.action_start_frame), y0 + 4),
            (xpos(sample.action_end_frame), y0 + bar_h - 4),
            (60, 170, 60),
            -1,
        )

    cv2.line(
        img,
        (xpos(sample.annotation_frame), y0),
        (xpos(sample.annotation_frame), y0 + bar_h),
        (0, 220, 255),
        2,
    )

    for i, obj in enumerate(sample.objects[:3]):
        cf = obj.contact_frame(sample.annotation_frame, sample.fps)
        cv2.line(img, (xpos(cf), y0), (xpos(cf), y0 + bar_h), (80, 80, 255), 1)

    cv2.line(img, (xpos(current_frame), y0 - 4), (xpos(current_frame), y0 + bar_h + 4), (255, 255, 255), 2)

    labels = [
        (f"clip [{sample.clip_start_frame}, {sample.clip_end_frame}]", (200, 220, 255)),
        (f"STA annotation frame @ {sample.annotation_frame}", (0, 220, 255)),
    ]
    if sample.action_start_frame is not None:
        labels.insert(
            1,
            (
                f"action [{sample.action_start_frame}, {sample.action_end_frame}]",
                (120, 220, 120),
            ),
        )
    for i, (txt, color) in enumerate(labels):
        cv2.putText(img, txt, (x0, y0 - 8 - i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_gt_boxes(img, sample: StaSample, current_frame: int):
    if current_frame != sample.annotation_frame:
        return
    h, w = img.shape[:2]
    for obj in sample.objects:
        x1, y1, x2, y2 = obj.box
        if max(x1, x2) <= 1.0 and max(y1, y2) <= 1.0:
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 255, 120), 2)


def annotate_frame(
    img,
    sample: StaSample,
    frame_idx: int,
    sample_idx: int,
    n_samples: int,
    paused: bool,
):
    vis = img.copy()
    draw_gt_boxes(vis, sample, frame_idx)
    draw_timeline(vis, sample, frame_idx)

    obj_lines = []
    for j, obj in enumerate(sample.objects[:6]):
        ttc = remaining_ttc(obj, sample.annotation_frame, frame_idx, sample.fps)
        obj_lines.append(
            (
                f"obj{j + 1}: {obj.verb_name} + {obj.noun_name} | "
                f"TTC={ttc:.2f}s (gt @ ann: {obj.time_to_contact:.2f}s)",
                (120, 255, 120),
            )
        )

    draw_text_block(
        vis,
        [
            (f"[{sample_idx + 1}/{n_samples}] {sample.video_uid} ({sample.split})", (255, 255, 255)),
            (f"uid: {sample.uid}", (180, 180, 180)),
            (f"annotation frame: {sample.annotation_frame}  current: {frame_idx}", (255, 255, 255)),
            (f"video fps: {sample.fps:.3f}", (200, 200, 200)),
            *obj_lines,
            ("SPACE pause | n/p sample | arrows frame | q quit", (160, 160, 160)),
            ("PAUSED" if paused else "", (0, 180, 255)),
        ],
    )
    return vis


class VideoCache:
    def __init__(self, path: Path):
        self.reader = PyAVVideoReader(str(path))

    def load(self, frame_idx: int) -> np.ndarray | None:
        frames = self.reader[[max(0, frame_idx)]]
        return frames[0]


def load_frame(video: VideoCache, frame_idx: int, scale: int, max_width: int | None) -> np.ndarray:
    img = video.load(frame_idx)
    if img is None:
        placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            f"missing frame {frame_idx}",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return placeholder
    return fit_display(img, scale=scale, max_width=max_width)


def step_frame(frame_pos: int, delta: int, n_frames: int) -> int:
    return max(0, min(frame_pos + delta, n_frames - 1))


def play_sample(
    sample: StaSample,
    sample_idx: int,
    n_samples: int,
    wait_ms: int,
    scale: int,
    max_width: int | None,
):
    paused = False
    frame_pos = 0
    window = "Ego4D STA (GT)"
    video = VideoCache(sample.mp4)

    while True:
        indices = sample.playback_frames
        frame_idx = indices[frame_pos]
        img = load_frame(video, frame_idx, scale=scale, max_width=max_width)
        vis = annotate_frame(img, sample, frame_idx, sample_idx, n_samples, paused)
        cv2.imshow(window, vis)
        delay = 0 if paused else wait_ms
        key = cv2.waitKeyEx(delay)
        if key == -1:
            if not paused:
                frame_pos += 1
                if frame_pos >= len(indices):
                    frame_pos = 0
            continue

        if key in (ord("q"), 27):
            return "quit"
        if key == ord(" "):
            paused = not paused
        elif key == ord("n"):
            return "next"
        elif key == ord("p"):
            return "prev"
        elif key in KEY_RIGHT or key in (ord("]"), ord(".")):
            frame_pos = step_frame(frame_pos, 1, len(indices))
        elif key in KEY_LEFT or key in (ord("["), ord(",")):
            frame_pos = step_frame(frame_pos, -1, len(indices))
        elif not paused:
            frame_pos += 1
            if frame_pos >= len(indices):
                frame_pos = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["local", "remote"], default="local")
    parser.add_argument(
        "--remote_root",
        default=None,
        help="Override remote dataset root (path or sftp URL)",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--video_uid", default=None, help="Only visualize this video_uid")
    parser.add_argument("--fps", type=float, default=8.0, help="Playback subsampling FPS")
    parser.add_argument("--wait_ms", type=int, default=int(1000 / 8), help="Delay between frames")
    parser.add_argument("--scale", type=int, default=DISPLAY_SCALE, help="Display upscale factor")
    parser.add_argument(
        "--max_width",
        type=int,
        default=DEFAULT_MAX_DISPLAY_WIDTH,
        help="Max display width in pixels (0 = no limit)",
    )
    parser.add_argument(
        "--include_unannotated",
        action="store_true",
        help="Include test entries without object GT",
    )
    parser.add_argument("--headless_check", action="store_true", help="Write gt_preview.jpg and exit")
    args = parser.parse_args()

    display_scale = max(1, args.scale)
    max_width = args.max_width if args.max_width > 0 else None
    data_root = resolve_data_root(args.source, args.remote_root)
    if not data_root.is_dir():
        print(f"Data root does not exist: {data_root}", file=sys.stderr)
        sys.exit(1)

    samples = build_sta_samples(
        data_root,
        split=args.split,
        target_fps=args.fps,
        require_objects=not args.include_unannotated,
        require_video=True,
        video_uid=args.video_uid,
    )
    if not samples:
        print(f"No STA samples with local video under {data_root}", file=sys.stderr)
        sys.exit(1)

    print(f"source={args.source} root={data_root} samples={len(samples)} playback_fps={args.fps}")

    if args.headless_check:
        s = samples[0]
        video = VideoCache(s.mp4)
        frame_idx = s.annotation_frame
        img = load_frame(video, frame_idx, scale=display_scale, max_width=max_width)
        out = annotate_frame(img, s, frame_idx, 0, len(samples), False)
        cv2.imwrite("gt_preview.jpg", out)
        print(f"Wrote gt_preview.jpg ({out.shape[1]}x{out.shape[0]})")
        return

    idx = 0
    while 0 <= idx < len(samples):
        action = play_sample(
            samples[idx], idx, len(samples), wait_ms=args.wait_ms, scale=display_scale, max_width=max_width
        )
        if action == "quit":
            break
        if action == "next":
            idx += 1
        elif action == "prev":
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
