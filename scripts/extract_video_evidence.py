#!/usr/bin/env python3
"""Extract metadata, uniformly sampled frames, and a contact sheet from a video."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract reproducible evidence for reference-video analysis. "
            "Requires ffmpeg and ffprobe."
        )
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--output", required=True, type=Path, help="Empty output directory")
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="Number of uniformly spaced frames (default: 12)",
    )
    parser.add_argument("--ffmpeg", help="Path or command name for ffmpeg")
    parser.add_argument("--ffprobe", help="Path or command name for ffprobe")
    return parser.parse_args()


def executable(explicit: str | None, default: str) -> str:
    candidate = explicit or default
    found = shutil.which(candidate)
    if found:
        return found
    path = Path(candidate)
    if path.is_file():
        return str(path.resolve())
    raise RuntimeError(
        f"Cannot find {default}. Install FFmpeg or pass --{default} with its path."
    )


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def video_stream(probe: dict[str, Any]) -> dict[str, Any]:
    stream = next(
        (
            item
            for item in probe.get("streams", [])
            if item.get("codec_type") == "video"
        ),
        None,
    )
    if stream is None:
        raise RuntimeError("The input contains no video stream.")
    return stream


def duration(probe: dict[str, Any], stream: dict[str, Any]) -> float:
    for value in (stream.get("duration"), probe.get("format", {}).get("duration")):
        if value not in (None, "N/A") and float(value) > 0:
            return float(value)
    raise RuntimeError("Unable to determine a positive video duration.")


def frame_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" not in value:
        return float(value)
    numerator, denominator = map(float, value.split("/", 1))
    return numerator / denominator if denominator else None


def times_for(video_duration: float, count: int) -> list[float]:
    safe_end = max(0.0, video_duration - min(0.05, video_duration / 100.0))
    return [safe_end * index / (count - 1) for index in range(count)]


def extract_frame(
    ffmpeg: str,
    source: Path,
    timestamp: float,
    destination: Path,
) -> None:
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(destination),
        ]
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not create frame: {destination}")


def contact_sheet(ffmpeg: str, frames: Path, destination: Path, count: int) -> None:
    columns = min(4, count)
    rows = math.ceil(count / columns)
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "1",
            "-start_number",
            "1",
            "-i",
            str(frames / "frame_%03d.jpg"),
            "-vf",
            f"scale=480:-2,tile={columns}x{rows}:padding=6:margin=6:color=white",
            "-frames:v",
            "1",
            "-y",
            str(destination),
        ]
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("FFmpeg did not create the contact sheet.")


def main() -> int:
    args = arguments()
    try:
        source = args.video.expanduser().resolve(strict=True)
        if not source.is_file():
            raise RuntimeError(f"Input is not a file: {source}")
        if not 2 <= args.samples <= 120:
            raise RuntimeError("--samples must be between 2 and 120.")

        ffmpeg = executable(args.ffmpeg, "ffmpeg")
        ffprobe = executable(args.ffprobe, "ffprobe")
        output = args.output.expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(
                "Output directory must be empty to avoid mixing evidence from different runs."
            )
        output.mkdir(parents=True, exist_ok=True)
        frames = output / "keyframes"
        frames.mkdir()

        probe = json.loads(
            run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(source),
                ]
            )
        )
        stream = video_stream(probe)
        video_duration = duration(probe, stream)
        timestamps = times_for(video_duration, args.samples)
        manifest: list[dict[str, Any]] = []

        for index, timestamp in enumerate(timestamps, 1):
            filename = f"frame_{index:03d}.jpg"
            extract_frame(ffmpeg, source, timestamp, frames / filename)
            manifest.append(
                {
                    "index": index,
                    "timestamp_seconds": round(timestamp, 6),
                    "file": f"keyframes/{filename}",
                }
            )

        shutil.copy2(frames / "frame_001.jpg", output / "first-frame.jpg")
        shutil.copy2(
            frames / f"frame_{args.samples:03d}.jpg",
            output / "last-frame.jpg",
        )
        contact_sheet(ffmpeg, frames, output / "contact-sheet.jpg", args.samples)
        metadata = {
            "source": str(source),
            "duration_seconds": round(video_duration, 6),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "codec": stream.get("codec_name"),
            "pixel_format": stream.get("pix_fmt"),
            "average_fps": frame_rate(stream.get("avg_frame_rate")),
            "nominal_fps": frame_rate(stream.get("r_frame_rate")),
            "sample_count": args.samples,
            "samples": manifest,
            "probe": probe,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Evidence written to: {output}")
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
