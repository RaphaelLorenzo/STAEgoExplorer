#!/usr/bin/env python3
"""
Exploration stats for Ego4D Short-Term Action (STA) annotations (fho_sta_<split>.json).

Schema: https://ego4d-data.org/docs/benchmarks/forecasting/
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

SPLIT_COLORS = {"train": "C0", "val": "C1"}


def hist_weights(values: list[float]) -> list[float]:
    n = len(values)
    return [100.0 / n] * n if n else []


def summarize_floats(
    label: str,
    values: list[float],
    hist_bins: int = 10,
    plot_path: Path | None = None,
) -> None:
    if not values:
        print(f"\n{label}: (no values)")
        return
    values_sorted = sorted(values)
    print(f"\n{label} (n={len(values)})")
    print(
        f"  min={values_sorted[0]:.3f}  max={values_sorted[-1]:.3f}  "
        f"mean={statistics.mean(values):.3f}  stdev={statistics.pstdev(values):.3f}  "
        f"median={statistics.median(values):.3f}"
    )
    if len(values_sorted) >= 4:
        q1, q2, q3 = statistics.quantiles(values, n=4)
        print(f"  quartiles: Q1={q1:.3f}  Q2={q2:.3f}  Q3={q3:.3f}")

    lo, hi = values_sorted[0], values_sorted[-1]
    if lo == hi:
        print(f"  histogram: all values = {lo:.3f}")
    else:
        width = (hi - lo) / hist_bins
        counts = [0] * hist_bins
        for v in values:
            idx = min(int((v - lo) / width), hist_bins - 1)
            counts[idx] += 1
        print(f"  histogram ({hist_bins} bins, width={width:.3f}):")
        max_count = max(counts)
        for i, c in enumerate(counts):
            left = lo + i * width
            right = left + width
            bar_len = int(40 * c / max_count) if max_count else 0
            print(f"    [{left:7.2f}, {right:7.2f}): {c:6d}  {'#' * bar_len}")

    if plot_path is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(values, bins=hist_bins, weights=hist_weights(values), label=f"n={len(values)}")
        ax.set_title(label)
        ax.set_ylabel("%")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  plot: {plot_path}")


def summarize_integer_counts(
    label: str,
    values: list[int],
    plot_path: Path | None = None,
) -> None:
    if not values:
        print(f"\n{label}: (no values)")
        return
    counts = Counter(values)
    lo, hi = min(values), max(values)
    print(f"\n{label} (n={len(values)})")
    print(
        f"  min={lo}  max={hi}  mean={statistics.mean(values):.3f}  "
        f"median={statistics.median(values):.3f}"
    )
    print("  counts per integer:")
    max_count = max(counts.values())
    for k in range(lo, hi + 1):
        c = counts.get(k, 0)
        bar_len = int(40 * c / max_count) if max_count else 0
        print(f"    {k:3d}: {c:6d}  {'#' * bar_len}")

    if plot_path is not None:
        total = len(values)
        xs = list(range(lo, hi + 1))
        ys = [100.0 * counts.get(x, 0) / total for x in xs]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(xs, ys, width=0.8, align="center", label=f"n={total}")
        ax.set_xticks(xs)
        ax.set_title(label)
        ax.set_xlabel("objects per annotation")
        ax.set_ylabel("% of annotations")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  plot: {plot_path}")


def print_class_counts(
    label: str,
    counts: Counter,
    id_to_name: dict[int, str],
    top_k: int,
    plot_path: Path | None = None,
) -> None:
    print(f"\n{label} (unique classes with ≥1 sample: {len(counts)} / {len(id_to_name)} taxonomy)")
    print(f"  total object samples: {sum(counts.values())}")
    top = counts.most_common(top_k)
    for cid, n in top:
        name = id_to_name.get(cid, "?")
        print(f"    {cid:3d} {name:30s} {n:6d}")
    if len(counts) > top_k:
        print(f"    ... ({len(counts) - top_k} more classes)")

    if plot_path is not None and top:
        total = sum(counts.values())
        labels = [f"{cid}: {id_to_name.get(cid, '?')[:28]}" for cid, _ in top]
        vals = [100.0 * n / total for _, n in top]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(top))))
        ax.barh(
            range(len(top)),
            vals,
            label=f"n={total}, {len(counts)}/{len(id_to_name)} classes, showing {len(top)}",
        )
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(label)
        ax.set_xlabel("% of object samples")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  plot: {plot_path}")


def shared_hist_bins(values_a: list[float], values_b: list[float], hist_bins: int) -> list[float]:
    lo = min(min(values_a), min(values_b))
    hi = max(max(values_a), max(values_b))
    if lo == hi:
        return [lo, hi + 1e-9]
    step = (hi - lo) / hist_bins
    return [lo + i * step for i in range(hist_bins + 1)]


def class_legend_label(split: str, counts: Counter, taxonomy_size: int, n_shown: int) -> str:
    return (
        f"{split} (n={sum(counts.values())}, "
        f"{len(counts)}/{taxonomy_size} classes, showing {n_shown})"
    )


def plot_overlaid_histograms(
    title: str,
    series: dict[str, list[float]],
    plot_path: Path,
    hist_bins: int = 10,
    xlabel: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    all_vals = [v for vals in series.values() for v in vals]
    bins = shared_hist_bins(series[list(series.keys())[0]], series[list(series.keys())[1]], hist_bins)
    if len(all_vals) == 1 or bins[0] == bins[-1]:
        bins = hist_bins
    for split, values in series.items():
        ax.hist(
            values,
            bins=bins,
            weights=hist_weights(values),
            alpha=0.55,
            label=f"{split} (n={len(values)})",
            color=SPLIT_COLORS.get(split, None),
        )
    ax.set_title(title)
    ax.set_ylabel("%")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  plot: {plot_path}")


def plot_grouped_integer_bars(
    title: str,
    series: dict[str, list[int]],
    plot_path: Path,
) -> None:
    counters = {split: Counter(vals) for split, vals in series.items()}
    lo = min(min(vals) for vals in series.values())
    hi = max(max(vals) for vals in series.values())
    xs = list(range(lo, hi + 1))
    n = len(series)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (split, counter) in enumerate(counters.items()):
        total = len(series[split])
        offset = (i - (n - 1) / 2) * width
        ys = [100.0 * counter.get(x, 0) / total for x in xs]
        ax.bar(
            [x + offset for x in xs],
            ys,
            width=width,
            label=f"{split} (n={total})",
            color=SPLIT_COLORS.get(split, None),
        )
    ax.set_xticks(xs)
    ax.set_title(title)
    ax.set_xlabel("objects per annotation")
    ax.set_ylabel("% of annotations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  plot: {plot_path}")


def plot_grouped_class_counts(
    title: str,
    counts_by_split: dict[str, Counter],
    id_to_name: dict[int, str],
    taxonomy_size: int,
    top_k: int,
    plot_path: Path,
) -> None:
    merged: Counter = Counter()
    for c in counts_by_split.values():
        merged.update(c)
    top = merged.most_common(top_k)
    if not top:
        return
    top_ids = [cid for cid, _ in top]
    n_shown = len(top_ids)
    labels = [f"{cid}: {id_to_name.get(cid, '?')[:28]}" for cid in top_ids]

    splits = list(counts_by_split.keys())
    n_splits = len(splits)
    bar_h = 0.8 / n_splits
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * n_shown)))

    for i, split in enumerate(splits):
        counts = counts_by_split[split]
        total = sum(counts.values())
        vals = [100.0 * counts.get(cid, 0) / total for cid in top_ids] if total else [0] * n_shown
        y_pos = [j + (i - (n_splits - 1) / 2) * bar_h for j in range(n_shown)]
        ax.barh(
            y_pos,
            vals,
            height=bar_h,
            label=class_legend_label(split, counts, taxonomy_size, n_shown),
            color=SPLIT_COLORS.get(split, None),
        )

    ax.set_yticks(range(n_shown))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("% of object samples")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  plot: {plot_path}")


def collect_stats(path: Path, data: dict) -> dict:
    info = data.get("info", {})
    anns = data["annotations"]
    verb_name = {c["id"]: c["name"] for c in data.get("verb_categories", [])}
    noun_name = {c["id"]: c["name"] for c in data.get("noun_categories", [])}
    split = info.get("split") or path.stem.replace("fho_sta_", "")

    samples_per_video = Counter(a.get("video_uid") or a.get("video_id") for a in anns)
    objects_per_ann = [len(a.get("objects") or []) for a in anns]
    verb_counts: Counter = Counter()
    noun_counts: Counter = Counter()
    ttc_values: list[float] = []
    for a in anns:
        for obj in a.get("objects") or []:
            verb_counts[obj["verb_category_id"]] += 1
            noun_counts[obj["noun_category_id"]] += 1
            ttc_values.append(float(obj["time_to_contact"]))

    return {
        "split": split,
        "verb_name": verb_name,
        "noun_name": noun_name,
        "samples_per_video": [float(n) for n in samples_per_video.values()],
        "objects_per_ann": objects_per_ann,
        "verb_counts": verb_counts,
        "noun_counts": noun_counts,
        "ttc_values": ttc_values,
    }


def plot_trainval(train: dict, val: dict, plot_dir: Path, top_k: int) -> None:
    out = plot_dir / "trainval"
    out.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 72)
    print(f"Combined train+val plots -> {out}")

    plot_overlaid_histograms(
        "annotation records per video",
        {"train": train["samples_per_video"], "val": val["samples_per_video"]},
        out / "samples_per_video.png",
    )

    if train["objects_per_ann"] and val["objects_per_ann"]:
        plot_grouped_integer_bars(
            "objects per annotation record",
            {"train": train["objects_per_ann"], "val": val["objects_per_ann"]},
            out / "objects_per_annotation.png",
        )
        plot_grouped_class_counts(
            "verb classes (object samples)",
            {"train": train["verb_counts"], "val": val["verb_counts"]},
            train["verb_name"],
            len(train["verb_name"]),
            top_k,
            out / "verb_classes_top.png",
        )
        plot_grouped_class_counts(
            "noun classes (object samples)",
            {"train": train["noun_counts"], "val": val["noun_counts"]},
            train["noun_name"],
            len(train["noun_name"]),
            top_k,
            out / "noun_classes_top.png",
        )
        plot_overlaid_histograms(
            "time_to_contact (sec, per object sample)",
            {"train": train["ttc_values"], "val": val["ttc_values"]},
            out / "time_to_contact.png",
            xlabel="seconds",
        )


def process_file(path: Path, top_k: int, plot_dir: Path) -> dict:
    print("=" * 72)
    print(path)
    with path.open() as f:
        data = json.load(f)

    stats = collect_stats(path, data)
    split = stats["split"]
    anns = data["annotations"]
    verb_name = stats["verb_name"]
    noun_name = stats["noun_name"]
    info = data.get("info", {})
    out = plot_dir / split
    out.mkdir(parents=True, exist_ok=True)

    print(f"split: {split}  version: {info.get('version')}  "
          f"include_annotations: {info.get('include_annotations')}")
    print(f"annotation records (frame-level): {len(anns)}")

    clip_uids = {a["clip_uid"] for a in anns}
    video_uids = {a.get("video_uid") or a.get("video_id") for a in anns}
    print(f"unique clips (clip_uid): {len(clip_uids)}")
    print(f"unique videos: {len(video_uids)}")
    if video_uids:
        print(f"avg annotation records per video: {len(anns) / len(video_uids):.2f}")

    summarize_floats(
        "annotation records per video",
        stats["samples_per_video"],
        plot_path=out / "samples_per_video.png",
    )

    if any(stats["objects_per_ann"]):
        summarize_integer_counts(
            "objects per annotation record",
            stats["objects_per_ann"],
            plot_path=out / "objects_per_annotation.png",
        )
        print_class_counts(
            "verb classes (object samples)",
            stats["verb_counts"],
            verb_name,
            top_k,
            plot_path=out / "verb_classes_top.png",
        )
        print_class_counts(
            "noun classes (object samples)",
            stats["noun_counts"],
            noun_name,
            top_k,
            plot_path=out / "noun_classes_top.png",
        )
        summarize_floats(
            "time_to_contact (sec, per object sample)",
            stats["ttc_values"],
            plot_path=out / "time_to_contact.png",
        )
    else:
        print("\n(no 'objects' entries — e.g. test unannotated split; skipping class / TTC stats)")

    return stats


def main() -> None:
    default_dir = Path(__file__).resolve().parents[1] / "data/ego4d_data/v2/annotations"
    default_plot_dir = Path(__file__).resolve().parents[1] / "output/sta_annotation_stats"
    parser = argparse.ArgumentParser(description="Stats for Ego4D fho_sta_*.json files")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"Annotation JSON paths (default: {default_dir}/fho_sta_*.json)",
    )
    parser.add_argument("--top-k", type=int, default=25, help="Top classes to print and plot per taxonomy")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=default_plot_dir,
        help="Directory for saved plots (subfolder per split)",
    )
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        paths = sorted(default_dir.glob("fho_sta_*.json"))
    if not paths:
        parser.error("no annotation files found")

    stats_by_split: dict[str, dict] = {}
    for path in paths:
        if not path.is_file():
            parser.error(f"not a file: {path}")
        stats = process_file(path, args.top_k, args.plot_dir)
        stats_by_split[stats["split"]] = stats

    train = stats_by_split.get("train")
    val = stats_by_split.get("val")
    if (
        train
        and val
        and train["ttc_values"]
        and val["ttc_values"]
    ):
        plot_trainval(train, val, args.plot_dir, args.top_k)


if __name__ == "__main__":
    main()
