# STAEgoExplorer

Exploration tools for [Ego4D Short-Term Action (STA)](https://ego4d-data.org/docs/benchmarks/forecasting/) annotations: dataset statistics, plots, and an interactive ground-truth video visualizer.

The vendored [`forecasting/`](forecasting/) tree is the official Ego4D forecasting baseline (reference for annotation fields and `PyAVVideoReader` behavior). This repo does not require installing that package for the tools below.

## Data layout

Expected v2 layout (local default: `data/ego4d_data/v2/`):

```
data/ego4d_data/v2/
  annotations/
    fho_sta_train.json
    fho_sta_val.json
    fho_sta_test_unannotated.json
  full_scale/
    {video_uid}.mp4
```

Download with the [Ego4D CLI](https://github.com/facebookresearch/Ego4d), e.g. full-scale videos and FHO annotations:

```bash
python -m ego4d.cli.cli --output_directory="~/ego4d_data" --datasets full_scale annotations --benchmarks FHO --version v2
```

Then symlink or copy into `data/ego4d_data/v2/` as above.

### Local vs Jean Zay (remote)

| Mode | Root |
|------|------|
| **local** (default) | `data/ego4d_data/v2/` |
| **remote** | `/lustre/fsstor/projects/rech/lfh/ufg41mh/ego4d_data/v2` |

Remote SFTP URL (mapped to the lustre path when mounted on Jean Zay):

`sftp://jean-zay/lustre/fsstor/projects/rech/lfh/ufg41mh/ego4d_data/v2/`

Override with `--remote_root` or environment variable `EGO4D_DATA_ROOT` (path or `sftp://…` URL).

---

## Annotation statistics (`scripts/sta_annotation_stats.py`)

Prints summary statistics to the terminal and saves matplotlib figures. Targets **`fho_sta_<split>.json`** ([schema](https://ego4d-data.org/docs/benchmarks/forecasting/)).

### Install

```bash
pip install matplotlib
```

### Usage

```bash
# All fho_sta_*.json under data/ego4d_data/v2/annotations/
python scripts/sta_annotation_stats.py

# One split
python scripts/sta_annotation_stats.py data/ego4d_data/v2/annotations/fho_sta_val.json

# Options
python scripts/sta_annotation_stats.py --top-k 30 --plot-dir output/sta_annotation_stats
```

### Output

**Console:** unique clips/videos, frame-level annotation counts, samples per video, objects per annotation (integer histogram), verb/noun class counts (with taxonomy names), time-to-contact distribution.

**Plots** under `output/sta_annotation_stats/<split>/` (percentages on axes; totals in legends):

| File | Description |
|------|-------------|
| `samples_per_video.png` | Distribution of annotation records per video |
| `objects_per_annotation.png` | Integer counts (1, 2, … objects per frame record) |
| `verb_classes_top.png` / `noun_classes_top.png` | Top‑k classes (% of object samples) |
| `time_to_contact.png` | TTC histogram |

When **train** and **val** are processed in the same run, combined plots are written to **`output/sta_annotation_stats/trainval/`** with train (blue) vs val (orange). Class plots use the top‑k classes from merged counts; legends include `n=…`, `classes with samples / taxonomy size`, and `showing k`.

Splits without `objects` (e.g. `fho_sta_test_unannotated.json`) skip class/TTC stats.

---

## Ground-truth visualizer (`visualize_gt.py`)

Interactive OpenCV viewer (same general layout as ActionAnticipationExplorer’s `visualize_gt.py`) for STA samples with object annotations.

- Reads **`full_scale/{video_uid}.mp4`** via **PyAV** (`explorer/pyav_reader.py`, same frame indexing as the baseline dataloader).
- **Yellow timeline marker** at the STA annotation field **`frame`**.
- On-screen **verb + noun** and **time to contact** (counts down while playing forward from the annotation frame).
- GT **boxes** at the annotation frame; green bar = action span when present.

### Install

```bash
pip install -r requirements-visualize.txt
```

### Usage

```bash
# Local data, validation split (default)
python visualize_gt.py

python visualize_gt.py --split val --video_uid <uuid>

# Playback ~8 FPS over clip_parent span (default)
python visualize_gt.py --fps 8 --wait_ms 125

# Jean Zay lustre
python visualize_gt.py --source remote
python visualize_gt.py --source remote \
  --remote_root 'sftp://jean-zay/lustre/fsstor/projects/rech/lfh/ufg41mh/ego4d_data/v2/'

# Smoke test without GUI
python visualize_gt.py --headless_check   # writes gt_preview.jpg
```

Only samples whose MP4 exists under `full_scale/` are listed (`require_video=True`).

### Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `n` / `p` | Next / previous sample |
| Arrow keys, `[` `]` | Step frames |
| `q` | Quit |

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--source` | `local` | `local` or `remote` |
| `--split` | `val` | `train`, `val`, `test`, or `all` |
| `--fps` | `8` | Subsample native video FPS for playback |
| `--max_width` | `1280` | Downscale wide full-scale frames for display (`0` = no limit) |
| `--scale` | `1` | Extra integer upscale after max-width |
| `--include_unannotated` | off | Include test entries without object GT |

### Related code

- `explorer/ego4d_sta.py` — load annotations, build playback frame lists.
- `explorer/paths.py` — data roots and paths.
- Baseline dataloader: `forecasting/ego4d_forecasting/datasets/short_term_anticipation.py` (`frame`, `objects[]`, `time_to_contact`, etc.).

---

## Repository map

```
scripts/sta_annotation_stats.py   # STA JSON statistics + plots
visualize_gt.py                   # Interactive GT viewer
explorer/                         # Paths, STA loading, PyAV reader
requirements-visualize.txt        # deps for visualize_gt.py
forecasting/                      # Ego4D forecasting baseline (reference)
data/                             # Local Ego4D v2 data (gitignored)
output/sta_annotation_stats/      # Default plot output from sta_annotation_stats.py
```
