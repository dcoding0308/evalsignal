"""Download the public datasets used for case studies, with a manifest.

    uv run python audit/download_data.py

* lmsys/mt_bench_human_judgments (CC-BY-4.0): expert and GPT-4 judgments of
  the same model answers, for judge-reliability analysis.
* open-llm-leaderboard-old details for Mistral-7B-v0.1 and
  Mistral-7B-Instruct-v0.1 (no licence stated): per-question MMLU and ARC
  results, used only locally to compute statistics; never redistributed.

Each file's URL, size, SHA-256 and retrieval time go into data/raw/MANIFEST.json.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HF = "https://huggingface.co"
LEADERBOARD = {
    "mistral-7b-base": ("mistralai__Mistral-7B-v0.1", "2023-09-27T15-30-59.039834"),
    "mistral-7b-instruct": ("mistralai__Mistral-7B-Instruct-v0.1", "2023-10-10T06-38-48.353025"),
}


def _get(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                body: bytes = response.read()
                return body
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def _files(dataset: str) -> list[tuple[str, int]]:
    meta = json.loads(_get(f"{HF}/api/datasets/{dataset}?blobs=true"))
    return [(s["rfilename"], s.get("size") or 0) for s in meta["siblings"]]


def plan() -> list[tuple[str, str, Path]]:
    """(dataset, file in dataset, local path) for everything to download."""
    wanted = []
    for name, _ in _files("lmsys/mt_bench_human_judgments"):
        if name.endswith(".parquet"):
            wanted.append(
                ("lmsys/mt_bench_human_judgments", name, RAW / "mt_bench" / Path(name).name)
            )
    for label, (model, run) in LEADERBOARD.items():
        dataset = f"open-llm-leaderboard-old/details_{model}"
        for name, _ in _files(dataset):
            task = name.split("|")[1] if "|" in name else ""
            if name.startswith(run) and (
                task.startswith("hendrycksTest-") or task == "arc:challenge"
            ):
                wanted.append((dataset, name, RAW / "leaderboard" / label / f"{task}.parquet"))
    return wanted


def download(item: tuple[str, str, Path]) -> dict[str, object]:
    dataset, name, target = item
    url = f"{HF}/datasets/{dataset}/resolve/main/{urllib.parse.quote(name)}"
    data = target.read_bytes() if target.exists() else _get(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {
        "dataset": dataset,
        "file": name,
        "local": str(target.relative_to(RAW)),
        "url": url,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main() -> None:
    items = plan()
    print(f"{len(items)} files to fetch")
    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(download, items))
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "MANIFEST.json").write_text(
        json.dumps(sorted(entries, key=lambda e: str(e["local"])), indent=2)
    )
    print(f"downloaded {sum(int(str(e['bytes'])) for e in entries) / 1e6:.1f} MB; manifest written")


if __name__ == "__main__":
    main()
