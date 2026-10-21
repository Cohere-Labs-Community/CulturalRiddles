from __future__ import annotations

from pathlib import Path

from .utils import append_jsonl, ensure_dir, write_json


class RunWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        ensure_dir(self.run_dir)
        self.results_path = self.run_dir / "results.jsonl"
        self.run_config_path = self.run_dir / "run_config.json"
        self.run_metadata_path = self.run_dir / "run_metadata.json"
        self.manifest_path = self.run_dir / "manifest.json"

    def write_run_config(self, data: dict) -> None:
        write_json(self.run_config_path, data)

    def write_run_metadata(self, data: dict) -> None:
        write_json(self.run_metadata_path, data)

    def append_result(self, data: dict) -> None:
        append_jsonl(self.results_path, [data])

    def write_manifest(self, data: dict) -> None:
        write_json(self.manifest_path, data)
