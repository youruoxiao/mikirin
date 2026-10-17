from __future__ import annotations

import hashlib
import http.client
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from mikirin.atomic import atomic_write_json
from mikirin.layout import ProjectLayout


MOEGIRL_COMMIT = "1ff222f8af5e399e230b9f88eb50995869ed22bf"
MOEGIRL_FILES = {
    "LICENSE": "8f852bfb0154fd88a98bf5b2c04e1a39b55b5beb",
    "README.md": "482c0d026ded2f4ad6dfb7e5f64054db099a086c",
    "bangumi/bgm2moegirl.json": "ce3124ff015e894e7349031dfdad2b6f4c145517",
    "bangumi/bgm_images_medium_mapped.json": "54af6d95a4991188b46110e909e3b75a8ee3fca2",
    "bangumi/bgm_info.json": "1c69c5b91c9fef0fbcdd640ff3683b367bf7786e",
    "bangumi/bgm_redirects_full.json": "27d24ae1c4402402bd220ad7461bcdd9f3180215",
    "bangumi/moegirl2bgm.json": "b9686a52cc993e663a28c885eae256f5b49f0c88",
    "moegirl/attr2article.json": "b944b0e8648bb7a43741abe153fb4f61e3b0a3a7",
    "moegirl/attr2char.json": "2a4009cdf3507fefac6d7e70de337e97d9d2cc35",
    "moegirl/attr_index.json": "4cb9a26192a4ab763ab080a7417ec329e8a18ac1",
    "moegirl/attrs.json": "5bc150737cc2ef6326df97db9fcf63f4d5772142",
    "moegirl/char2attr.json": "5cadcaee91ee8e8af6d53865962de99cb0e33b97",
    "moegirl/char2cv.json": "346c29c5291a6695310e0d407e1a8a35d6964b87",
    "moegirl/char2gender.json": "0d41c700d06ccaf5acdeee3649f06526b30e2fac",
    "moegirl/char2subject.json": "adfe7d47770c05f8fc2fb156e42c246f0406f8f8",
    "moegirl/char_index.json": "40fac52667bbc5195973ecdb771ab5621c36a4e1",
    "moegirl/cv2char.json": "ea6e2955db160d874cf5672d1c56ecd3915f013a",
    "moegirl/cv_index.json": "50a58117359d49ada8a957a38a8bdab10aab194c",
    "moegirl/extra_processed.json": "2a3688bd0d0c7135f576fcde7434d18b55a1d877",
    "moegirl/eye_color_attr.json": "d5ee49219576d4ce38cac61eae4629f60910d45a",
    "moegirl/female_attr.json": "ae1cab6a80dae27fa669133cf49834fad27ae64f",
    "moegirl/fundamental_attr.json": "3134bd0543c7bffd5818ab3e1fb61a9517871112",
    "moegirl/hair_color_attr.json": "6c0e8d2c9b41b415438094c1e0e10e3d86a343b8",
    "moegirl/male_attr.json": "2aa5d72e902247d5c7e35ce43c98067a08445835",
    "moegirl/nogender_attr.json": "02cf5e3dddb9ee0415e1daae3d176b29d72ad137",
    "moegirl/subject_index.json": "deb360d7cc6cb292d02d2404d961fb15aab38010",
    "moegirl/subjects.json": "6333757460e496461dd3da36b45c0cb63652ba93",
}

_RAW_BASE = (
    "https://ghproxy.net/https://raw.githubusercontent.com/"
    f"Zzzzzzyt/moegirl-dataset/{MOEGIRL_COMMIT}/"
)
_USER_AGENT = "MikirinArchive/0.1 (+local anime knowledge base)"


def git_blob_sha1_bytes(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def git_blob_sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MoegirlFileRecord:
    relative_path: str
    git_blob_sha1: str
    byte_size: int
    fetched_at: str


class MoegirlDatasetFetcher:
    def __init__(
        self,
        layout: ProjectLayout,
        *,
        max_attempts: int = 8,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.layout = layout
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds

    @property
    def root(self) -> Path:
        return self.layout.raw / "moegirl_dataset" / MOEGIRL_COMMIT

    @property
    def manifest_path(self) -> Path:
        return self.layout.manifests / "moegirl_dataset" / f"{MOEGIRL_COMMIT}.json"

    def _download_file(self, relative_path: str, expected_blob: str) -> MoegirlFileRecord:
        destination = self.root / Path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and git_blob_sha1_file(destination) == expected_blob:
            return MoegirlFileRecord(
                relative_path=destination.relative_to(self.layout.root).as_posix(),
                git_blob_sha1=expected_blob,
                byte_size=destination.stat().st_size,
                fetched_at=datetime.fromtimestamp(destination.stat().st_mtime, UTC).isoformat(),
            )
        partial = destination.with_name(f".{destination.name}.part")
        url = urllib.parse.urljoin(_RAW_BASE, relative_path)
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": _USER_AGENT, "Accept": "application/octet-stream"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    status = int(response.status)
                    mode = "ab" if offset and status == 206 else "wb"
                    with partial.open(mode) as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                if git_blob_sha1_file(partial) != expected_blob:
                    raise ValueError(f"Git blob SHA-1 不符: {relative_path}")
                os.replace(partial, destination)
                return MoegirlFileRecord(
                    relative_path=destination.relative_to(self.layout.root).as_posix(),
                    git_blob_sha1=expected_blob,
                    byte_size=destination.stat().st_size,
                    fetched_at=datetime.now(UTC).isoformat(),
                )
            except (
                OSError,
                urllib.error.URLError,
                http.client.IncompleteRead,
                ValueError,
            ) as error:
                last_error = error
                time.sleep(min(3 * 2 ** (attempt - 1), 60))
        raise RuntimeError(f"下载失败: {relative_path}") from last_error

    def fetch_all(self, *, limit: int | None = None) -> dict[str, object]:
        self.layout.initialize()
        records: dict[str, dict[str, object]] = {}
        failed: dict[str, str] = {}
        attempted = 0
        for relative_path, expected_blob in MOEGIRL_FILES.items():
            destination = self.root / relative_path
            if destination.exists() and git_blob_sha1_file(destination) == expected_blob:
                record = self._download_file(relative_path, expected_blob)
                records[relative_path] = asdict(record)
                continue
            if limit is not None and attempted >= limit:
                continue
            attempted += 1
            try:
                record = self._download_file(relative_path, expected_blob)
                records[relative_path] = asdict(record)
            except Exception as error:
                failed[relative_path] = f"{type(error).__name__}: {error}"
            atomic_write_json(
                self.manifest_path,
                {
                    "commit": MOEGIRL_COMMIT,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "expected_files": len(MOEGIRL_FILES),
                    "files": records,
                    "failed": failed,
                },
            )
        return {
            "commit": MOEGIRL_COMMIT,
            "expected": len(MOEGIRL_FILES),
            "complete": len(records),
            "failed": len(failed),
        }
