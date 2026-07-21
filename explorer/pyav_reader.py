"""PyAV frame reader (subset of ego4d_forecasting.datasets.short_term_anticipation)."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, List

import av
import numpy as np


def pts_difference_per_frame(fps: Fraction, time_base: Fraction) -> int:
    pt = (1 / fps) * (1 / time_base)
    assert pt.denominator == 1, "should be whole number"
    return int(pt)


def frame_index_to_pts(frame: int, start_pt: int, diff_per_frame: int) -> int:
    return start_pt + frame * diff_per_frame


def pts_to_time_seconds(pts: int, base: Fraction) -> Fraction:
    return pts * base


def _get_frames_pts(
    video_pts_set: List[int],
    container: av.container.Container,
    include_audio: bool,
    include_additional_audio_pts: int,
) -> Iterable[av.frame.Frame]:
    assert len(container.streams.video) == 1

    min_pts = min(video_pts_set)
    max_pts = max(video_pts_set)
    video_pts_set = set(video_pts_set)

    video_stream = container.streams.video[0]
    fps: Fraction = video_stream.average_rate
    video_base: Fraction = video_stream.time_base
    video_pt_diff = pts_difference_per_frame(fps, video_base)

    clip_start_sec = pts_to_time_seconds(min_pts, video_base)
    clip_end_sec = pts_to_time_seconds(max_pts, video_base)
    clip_end_sec += max(
        pts_to_time_seconds(include_additional_audio_pts, video_base), 1 / fps
    )

    streams_to_decode = {"video": 0}
    if (
        include_audio
        and container.streams.audio is not None
        and len(container.streams.audio) > 0
    ):
        assert len(container.streams.audio) == 1
        streams_to_decode["audio"] = 0
        audio_base: Fraction = container.streams.audio[0].time_base

    seek_pts = max(0, min_pts - 2 * video_pt_diff)
    container.seek(seek_pts, stream=video_stream)
    if "audio" in streams_to_decode:
        audio_stream = container.streams.audio[0]
        audio_seek_pts = int(seek_pts * video_base / audio_base)
        audio_stream.seek(audio_seek_pts)

    previous_video_pts = None
    previous_audio_pts = None

    for frame in container.decode(**streams_to_decode):
        if isinstance(frame, av.AudioFrame):
            assert include_audio
            assert previous_audio_pts is None or previous_audio_pts < frame.pts
            previous_audio_pts = frame.pts
            audio_time_sec = pts_to_time_seconds(frame.pts, audio_base)
            if clip_start_sec <= audio_time_sec < clip_end_sec:
                yield frame
            elif audio_time_sec >= clip_end_sec:
                break

        elif isinstance(frame, av.VideoFrame):
            video_time_sec = pts_to_time_seconds(frame.pts, video_base)
            if video_time_sec >= clip_end_sec:
                break
            assert previous_video_pts is None or previous_video_pts < frame.pts
            if frame.pts in video_pts_set:
                yield frame


def _get_frames(
    video_frames: List[int],
    container: av.container.Container,
    include_audio: bool,
    audio_buffer_frames: int = 0,
) -> Iterable[av.frame.Frame]:
    assert len(container.streams.video) == 1

    video_stream = container.streams.video[0]
    video_start: int = video_stream.start_time
    fps: Fraction = video_stream.average_rate
    video_base: Fraction = video_stream.time_base
    video_pt_diff = pts_difference_per_frame(fps, video_base)

    audio_buffer_pts = (
        frame_index_to_pts(audio_buffer_frames, 0, video_pt_diff)
        if include_audio
        else 0
    )

    sort_idx = np.argsort(video_frames)
    rev_sort_idx = np.argsort(sort_idx)
    video_frames = [video_frames[i] for i in sort_idx]

    time_pts_set = [
        frame_index_to_pts(f, video_start, video_pt_diff) for f in video_frames
    ]
    time_pts_set, sampling_idx = np.unique(time_pts_set, return_inverse=True)
    time_pts_set = [int(i) for i in time_pts_set]

    frames = list(
        _get_frames_pts(time_pts_set, container, include_audio, audio_buffer_pts)
    )
    frames = [frames[i] for i in sampling_idx]
    frames = [frames[i] for i in rev_sort_idx]
    assert len(frames) == len(video_frames)
    return frames


class PyAVVideoReader:
    def __init__(
        self,
        path_to_video,
        include_audio=False,
        audio_buffer_frames=0,
        height=None,
    ):
        self.path_to_video = path_to_video
        self.include_audio = include_audio
        self.audio_buffer_frames = audio_buffer_frames
        self.height = height

    def __getitem__(self, frame_list):
        if isinstance(frame_list, (int, float)):
            frame_list = [int(frame_list)]
        else:
            frame_list = [int(f) for f in frame_list]

        with av.open(self.path_to_video) as input_video:
            frames = _get_frames(
                frame_list,
                input_video,
                include_audio=self.include_audio,
                audio_buffer_frames=self.audio_buffer_frames,
            )
            frames = list(frames)
        frames = [f.to_ndarray(format="bgr24") if f is not None else None for f in frames]
        if self.height is not None:
            import imutils

            frames = [
                imutils.resize(f, height=self.height) if f is not None else None for f in frames
            ]
        return frames
