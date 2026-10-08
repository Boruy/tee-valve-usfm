"""Download the official USFM pretrained backbone from its README Google Drive link."""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


FILE_ID = "1KRwXZgYterH895Z8EpXpR1L1eSMMJo4q"
SOURCE = f"https://drive.google.com/uc?export=download&id={FILE_ID}"


class DownloadForm(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action = None
        self.fields = {}
        self.in_form = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "form" and values.get("id") == "download-form":
            self.in_form = True
            self.action = values.get("action")
        elif tag == "input" and self.in_form and values.get("type") == "hidden":
            self.fields[values["name"]] = values.get("value", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self.in_form = False


def download(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    with opener.open(SOURCE, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("text/html"):
            form = DownloadForm()
            form.feed(response.read().decode("utf-8"))
            if not form.action or form.fields.get("id") != FILE_ID:
                raise RuntimeError("Official Google Drive download confirmation form was not found")
            url = f"{form.action}?{urllib.parse.urlencode(form.fields)}"
        else:
            url = SOURCE
    marker = output.with_name(output.name + ".INCOMPLETE")
    marker.write_text("Download in progress; do not load this checkpoint.\n", encoding="utf-8")
    sha = hashlib.sha256()
    size = 0
    with opener.open(url, timeout=120) as response:
        if response.headers.get("Content-Type", "").startswith("text/html"):
            raise RuntimeError("Google Drive returned HTML instead of checkpoint bytes")
        expected = int(response.headers.get("Content-Length", "0"))
        with output.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
                sha.update(chunk)
                size += len(chunk)
                if size // (50 * 1024 * 1024) != (size - len(chunk)) // (50 * 1024 * 1024):
                    print(f"Downloaded {size / 1024**2:.0f} MiB", flush=True)
    if size < 300_000_000 or (expected and size != expected):
        raise RuntimeError(f"Incomplete USFM checkpoint: {size} bytes, expected {expected}")
    record = {"source": SOURCE, "bytes": size, "sha256": sha.hexdigest()}
    output.with_name(output.name + ".download.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    marker.unlink()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/USFM_latest.pth"))
    args = parser.parse_args()
    print(json.dumps(download(args.output), indent=2))


if __name__ == "__main__":
    main()
