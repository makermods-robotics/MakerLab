import logging
from typing import Any

from huggingface_hub import HfApi, whoami
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

logger = logging.getLogger(__name__)

HF_RESOLVE_BASE = "https://huggingface.co/datasets"


def list_user_datasets() -> list[dict[str, Any]]:
    try:
        info = whoami()
    except (LocalTokenNotFoundError, HfHubHTTPError, OSError):
        return []

    authors = [info["name"]] + [o["name"] for o in info.get("orgs", [])]
    api = HfApi()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for author in authors:
        try:
            for ds in api.list_datasets(author=author, limit=200):
                if ds.id in seen:
                    continue
                seen.add(ds.id)
                out.append({
                    "repo_id": ds.id,
                    "last_modified": ds.last_modified.isoformat() if ds.last_modified else None,
                    "private": bool(getattr(ds, "private", False)),
                })
        except HfHubHTTPError as e:
            logger.warning(f"list_datasets({author}) failed: {e}")

    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out


def _format_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60:02d}:{s % 60:02d}"
    if s < 86400:
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    days = s // 86400
    rem = s % 86400
    return f"{days}d {rem // 3600:02d}:{(rem % 3600) // 60:02d}"


def get_episode_list(repo_id: str) -> dict[str, Any]:
    meta = LeRobotDatasetMetadata(repo_id)
    fps = meta.fps
    eps = meta.episodes

    out: list[dict[str, Any]] = []
    if eps is None:
        return {"fps": fps, "episodes": []}

    for idx in range(len(eps)):
        row = eps[idx]
        length = int(row["length"])
        tasks = row["tasks"]
        if hasattr(tasks, "tolist"):
            tasks = tasks.tolist()
        duration = length / fps if fps else 0.0
        out.append({
            "episode_index": idx,
            "length": length,
            "tasks": list(tasks) if tasks is not None else [],
            "duration_seconds": duration,
            "duration_human": _format_duration(duration),
        })

    return {"fps": fps, "total_episodes": meta.total_episodes, "episodes": out}


def get_replay_assets(repo_id: str, episode: int) -> dict[str, Any]:
    meta = LeRobotDatasetMetadata(repo_id)
    if episode < 0 or episode >= meta.total_episodes:
        raise IndexError(f"Episode {episode} out of range (0..{meta.total_episodes - 1})")

    action_names = meta.features["action"]["names"]
    if isinstance(action_names, dict):
        action_names = action_names.get("motors") or next(iter(action_names.values()), [])
    joint_names = list(action_names)

    cameras = []
    for vid_key in meta.video_keys:
        rel_path = meta.get_video_file_path(episode, vid_key).as_posix()
        url = f"{HF_RESOLVE_BASE}/{repo_id}/resolve/main/{rel_path}"
        cameras.append({"key": vid_key, "url": url})

    row = meta.episodes[episode]
    num_frames = int(row["length"])

    return {
        "joint_names": joint_names,
        "cameras": cameras,
        "fps": meta.fps,
        "num_frames": num_frames,
    }
