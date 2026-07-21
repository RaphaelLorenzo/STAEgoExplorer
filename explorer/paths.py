import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

LOCAL_DATA_ROOT = REPO_ROOT / "data" / "ego4d_data" / "v2"
REMOTE_SFTP_URL = (
    "sftp://jean-zay/lustre/fsstor/projects/rech/lfh/ufg41mh/ego4d_data/v2/"
)
REMOTE_DATA_ROOT = Path(
    "/lustre/fsstor/projects/rech/lfh/ufg41mh/ego4d_data/v2"
)


def sftp_url_to_path(url: str) -> Path:
    """Map sftp://host/abs/path -> /abs/path (Jean Zay lustre mount)."""
    if not url.startswith("sftp://"):
        return Path(url)
    without_scheme = url[len("sftp://") :]
    _, _, remote_path = without_scheme.partition("/")
    return Path("/" + remote_path)


def resolve_data_root(source: str, remote_root: str | None = None) -> Path:
    if os.path.exists(source):
        return Path(source)
    source_lc = source.lower()
    if source_lc == "local":
        return LOCAL_DATA_ROOT
    if source_lc == "remote":
        if remote_root:
            return (
                sftp_url_to_path(remote_root)
                if remote_root.startswith("sftp://")
                else Path(remote_root)
            )
        env_root = os.environ.get("EGO4D_DATA_ROOT")
        if env_root:
            return (
                sftp_url_to_path(env_root)
                if env_root.startswith("sftp://")
                else Path(env_root)
            )
        return REMOTE_DATA_ROOT
    raise ValueError(f"Unknown data source {source!r}. Use 'local', 'remote', or a path.")


def video_path(data_root: Path, video_uid: str) -> Path:
    return data_root / "full_scale" / f"{video_uid}.mp4"


def annotation_path(data_root: Path, split: str) -> Path:
    return data_root / "annotations" / f"fho_sta_{split}.json"
