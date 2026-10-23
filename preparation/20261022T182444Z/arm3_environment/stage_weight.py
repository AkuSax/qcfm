"""Stage the one approved AIMNet2 artifact only when its local file is absent."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "weights" / "aimnet2_wb97m_d3_0.pt"
URL = "https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_wb97m_d3_0.pt"
SIZE = 8836941
SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"


def check(path: Path) -> None:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    if size != SIZE or digest.hexdigest() != SHA256:
        raise ValueError(f"Weight integrity failure: bytes={size}, sha256={digest.hexdigest()}")


DEST.parent.mkdir(parents=True, exist_ok=True)
if DEST.exists():
    check(DEST)
    print(f"Verified existing weight: {DEST}")
else:
    temporary = DEST.with_suffix(DEST.suffix + ".partial")
    request = urllib.request.Request(URL, headers={"User-Agent": "qgcf-aimnet-preparation/1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != SIZE:
            raise ValueError(f"Unexpected weight length: {length}")
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > SIZE:
                raise ValueError("Weight download exceeded the pinned size")
            out.write(chunk)
    check(temporary)
    temporary.replace(DEST)
    (ROOT / "weight_stage_receipt.json").write_text(json.dumps({
        "url": URL, "path": str(DEST), "bytes": SIZE, "sha256": SHA256,
        "retrieved": True
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Staged and verified pinned weight: {DEST}")
