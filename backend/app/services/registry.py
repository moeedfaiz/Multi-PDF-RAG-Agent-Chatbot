import json
from pathlib import Path
from typing import Dict, List
from typing import Iterable

def registry_path(app_data_dir: Path) -> Path:
    return app_data_dir / "registry.jsonl"

def load_records(app_data_dir: Path) -> List[Dict]:
    p = registry_path(app_data_dir)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            # skip corrupted line
            continue
    return out

def append_record(app_data_dir: Path, record: Dict) -> None:
    p = registry_path(app_data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def rewrite_records(app_data_dir: Path, records: Iterable[Dict]) -> None:
    p = registry_path(app_data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
