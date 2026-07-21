"""Load Ego4D Short-Term Anticipation (fho_sta) annotations for visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .paths import annotation_path, video_path


@dataclass
class StaObject:
    box: tuple[float, float, float, float]
    verb_id: int
    noun_id: int
    verb_name: str
    noun_name: str
    time_to_contact: float

    def contact_frame(self, annotation_frame: int, fps: float) -> int:
        return int(round(annotation_frame + self.time_to_contact * fps))


@dataclass
class StaSample:
    uid: str
    video_uid: str
    split: str
    annotation_frame: int
    fps: float
    clip_start_frame: int
    clip_end_frame: int
    action_start_frame: int | None
    action_end_frame: int | None
    objects: list[StaObject]
    playback_frames: list[int]
    mp4: Path
    frame_width: int
    frame_height: int


def _split_annotation_files(data_root: Path, split: str) -> list[Path]:
    if split == "all":
        return sorted((data_root / "annotations").glob("fho_sta_*.json"))
    return [annotation_path(data_root, split)]


def load_sta_json(paths: list[Path]) -> dict:
    videos: dict = {}
    annotations: list = []
    verb_name: dict[int, str] = {}
    noun_name: dict[int, str] = {}
    for path in paths:
        with path.open() as f:
            j = json.load(f)
        file_split = j.get("info", {}).get("split") or path.stem.replace("fho_sta_", "")
        videos.update(j.get("info", {}).get("video_metadata", {}))
        verb_name.update({c["id"]: c["name"] for c in j.get("verb_categories", [])})
        noun_name.update({c["id"]: c["name"] for c in j.get("noun_categories", [])})
        for ann in j["annotations"]:
            ann = dict(ann)
            ann["_split"] = file_split
            annotations.append(ann)
    return {
        "videos": videos,
        "annotations": annotations,
        "verb_name": verb_name,
        "noun_name": noun_name,
    }


def subsample_frame_indices(
    start: int,
    end: int,
    native_fps: float,
    target_fps: float,
) -> list[int]:
    step = max(1, round(native_fps / target_fps))
    return list(range(start, end + 1, step))


def build_sta_samples(
    data_root: Path,
    split: str = "val",
    target_fps: float = 8.0,
    require_objects: bool = True,
    require_video: bool = True,
    video_uid: str | None = None,
) -> list[StaSample]:
    paths = _split_annotation_files(data_root, split)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing annotation file(s): {missing}")

    blob = load_sta_json(paths)
    samples: list[StaSample] = []

    for ann in blob["annotations"]:
        if video_uid and ann.get("video_uid") != video_uid:
            continue
        if require_objects and not ann.get("objects"):
            continue

        vid = ann["video_uid"]
        mp4 = video_path(data_root, vid)
        if require_video and not mp4.is_file():
            continue

        meta = blob["videos"].get(vid)
        if not meta:
            continue
        native_fps = float(meta["fps"])
        ann_frame = int(ann["frame"])

        clip_start = int(ann.get("clip_parent_start_frame", ann_frame - int(4 * native_fps)))
        clip_end = int(ann.get("clip_parent_end_frame", ann_frame + int(2 * native_fps)))
        action_start = ann.get("action_start_frame")
        action_end = ann.get("action_end_frame")
        action_start = int(action_start) if action_start is not None else None
        action_end = int(action_end) if action_end is not None else None

        objs: list[StaObject] = []
        for o in ann.get("objects") or []:
            objs.append(
                StaObject(
                    box=tuple(o["box"]),
                    verb_id=int(o["verb_category_id"]),
                    noun_id=int(o["noun_category_id"]),
                    verb_name=blob["verb_name"].get(o["verb_category_id"], "?"),
                    noun_name=blob["noun_name"].get(o["noun_category_id"], "?"),
                    time_to_contact=float(o["time_to_contact"]),
                )
            )

        playback = subsample_frame_indices(clip_start, clip_end, native_fps, target_fps)
        if ann_frame not in playback:
            playback.append(ann_frame)
            playback.sort()

        samples.append(
            StaSample(
                uid=ann["uid"],
                video_uid=vid,
                split=ann.get("_split", split),
                annotation_frame=ann_frame,
                fps=native_fps,
                clip_start_frame=clip_start,
                clip_end_frame=clip_end,
                action_start_frame=action_start,
                action_end_frame=action_end,
                objects=objs,
                playback_frames=playback,
                mp4=mp4,
                frame_width=int(meta["frame_width"]),
                frame_height=int(meta["frame_height"]),
            )
        )

    return samples


def remaining_ttc(obj: StaObject, annotation_frame: int, current_frame: int, fps: float) -> float:
    elapsed = (current_frame - annotation_frame) / fps
    return max(0.0, obj.time_to_contact - elapsed)
