from __future__ import annotations

import http.client
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RangeSpec:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"非法字节范围: {self.start}-{self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def remaining(self, existing_size: int) -> tuple[int, int] | None:
        if existing_size < 0 or existing_size > self.length:
            raise ValueError(
                f"已有分片长度 {existing_size} 超过期望长度 {self.length}"
            )
        if existing_size == self.length:
            return None
        return self.start + existing_size, self.end


class HttpRangeFragmentFetcher:
    def __init__(
        self,
        *,
        max_attempts: int = 100,
        timeout_seconds: float = 90.0,
        retry_delay_seconds: float = 3.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.retry_delay_seconds = retry_delay_seconds

    def fetch(self, *, url: str, spec: RangeSpec, destination: Path) -> Path:
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            existing = destination.stat().st_size if destination.exists() else 0
            remaining = spec.remaining(existing)
            if remaining is None:
                return destination
            request_start, request_end = remaining
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "MikirinArchive/0.1 (+local anime knowledge base)",
                    "Accept": "application/octet-stream",
                    "Range": f"bytes={request_start}-{request_end}",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    if int(response.status) != 206:
                        raise RuntimeError(f"服务器未按 Range 返回 206: {response.status}")
                    content_range = response.headers.get("Content-Range", "")
                    expected_prefix = f"bytes {request_start}-"
                    if not content_range.startswith(expected_prefix):
                        raise RuntimeError(
                            f"Content-Range 起点不符: {content_range!r}, 期望 {expected_prefix!r}"
                        )
                    with destination.open("ab") as output:
                        bytes_left = spec.length - existing
                        try:
                            while chunk := response.read(1024 * 1024):
                                accepted = chunk[:bytes_left]
                                output.write(accepted)
                                output.flush()
                                bytes_left -= len(accepted)
                                if bytes_left == 0:
                                    break
                        except http.client.IncompleteRead as error:
                            if error.partial:
                                accepted = error.partial[:bytes_left]
                                output.write(accepted)
                                output.flush()
                            raise
                current_size = destination.stat().st_size
                if current_size > spec.length:
                    raise RuntimeError(
                        f"服务器返回超出请求范围的数据: {current_size}>{spec.length}"
                    )
                if current_size == spec.length:
                    return destination
            except (
                OSError,
                TimeoutError,
                ConnectionError,
                urllib.error.URLError,
                http.client.IncompleteRead,
                http.client.RemoteDisconnected,
                RuntimeError,
            ) as error:
                last_error = error
                if destination.exists() and destination.stat().st_size > spec.length:
                    raise
                if attempt < self.max_attempts:
                    time.sleep(min(self.retry_delay_seconds * attempt, 30.0))
        raise RuntimeError(
            f"分片在 {self.max_attempts} 次连接后仍未完整: {destination}"
        ) from last_error
