"""Content-addressed JSON cache. Same inputs -> same key -> no repeat API call."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def content_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class JsonCache:
    def __init__(self, root: Path, namespace: str, enabled: bool = True):
        self.dir = Path(root) / namespace
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path(key))
