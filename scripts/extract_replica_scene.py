"""Extract a small matching Replica scene subset from its split gzip tar release.

Usage: python scripts/extract_replica_scene.py frl_apartment_0
All split parts up to the requested mesh must already be downloaded.
"""

from __future__ import annotations

import argparse
import io
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "data/raw/SoundSpaces_Replica/replica_geometry_parts"
OUTPUT = ROOT / "data/raw/SoundSpaces_Replica/replica_geometry"
WANTED = {
    "mesh.ply",
    "habitat/mesh_semantic.ply",
    "habitat/mesh_semantic.navmesh",
    "habitat/info_semantic.json",
    "habitat/replica_stage.stage_config.json",
}


class SplitParts(io.RawIOBase):
    def __init__(self, parts: list[Path]):
        self.parts = iter(parts)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        view = memoryview(buffer)
        count = 0
        while count < len(view):
            if self.current is None:
                next_part = next(self.parts, None)
                if next_part is None:
                    break
                self.current = next_part.open("rb")
            got = self.current.readinto(view[count:])
            if got:
                count += got
            else:
                self.current.close()
                self.current = None
        return count

    def close(self):
        if self.current:
            self.current.close()
        super().close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--allow-incomplete", action="store_true", help="read through the first partial split part")
    args = parser.parse_args()
    scene = args.scene
    if "/" in scene or "\\" in scene or scene.startswith("."):
        parser.error("scene must be a single directory name")

    parts = sorted(PARTS.glob("replica_v1_0.tar.gz.part??"))
    if not parts:
        parser.error("no Replica release parts found")
    for index, part in enumerate(parts):
        expected = f"replica_v1_0.tar.gz.parta{chr(ord('a') + index)}"
        if part.name != expected:
            parser.error(f"missing preceding part before {part.name}")
        if part.stat().st_size != 2_000_000_000 and part.name != "replica_v1_0.tar.gz.partaq":
            parts = parts[:index + (1 if args.allow_incomplete else 0)]
            break

    found = set()
    target_root = OUTPUT / scene
    try:
        with SplitParts(parts) as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if name.startswith(scene + "/"):
                    relative = name[len(scene) + 1 :]
                    if relative in WANTED and member.isfile():
                        source = archive.extractfile(member)
                        assert source is not None
                        target = target_root / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix(target.suffix + ".partial")
                        with temporary.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                        temporary.replace(target)
                        found.add(relative)
                        print(f"extracted {relative}: {member.size} bytes", flush=True)
                elif found and not name.startswith(scene + "/"):
                    break
    except (EOFError, OSError, tarfile.TarError) as error:
        print(f"Archive needs additional complete parts: {error}", flush=True)
    print(f"Found {len(found)}/{len(WANTED)} selected files; used {len(parts)} part(s).", flush=True)


if __name__ == "__main__":
    main()
