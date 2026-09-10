"""
Burn subtitles into a video using ffmpeg (keeps the audio track intact --
a plain OpenCV re-encode would silently drop it, which matters for a
narrated demo video).

Captions are written in a simple plain-text format, not raw .srt, so you
don't have to hand-write timecodes for both a start and an end:

    captions.txt
    ------------
    # lines starting with # are comments, blank lines are ignored
    0:00 Hi, I'm Jeb -- this is my CT3205 real-time demo
    0:05 Starting webcam_demo.py with the new all-real model
    0:12-0:18 Holding up a card at an angle
    0:20 Even the tricky ones like "1" mostly work now

Each line is either:
    M:SS text                  (shown for --duration seconds, default 3,
                                 or until the next line starts -- whichever
                                 is sooner, so captions never overlap)
    M:SS-M:SS text              (explicit start-end)
Timestamps also accept H:MM:SS for anything over an hour.

Usage
-----
    python add_subtitles.py input.mp4 captions.txt output.mp4
    python add_subtitles.py input.mp4 captions.txt output.mp4 --font-size 32
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

TIME_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def parse_time(s):
    m = TIME_RE.match(s.strip())
    if not m:
        raise ValueError(f"Bad timestamp: {s!r} (expected M:SS or H:MM:SS)")
    h, mnt, sec = m.groups()
    h = int(h) if h else 0
    return h * 3600 + int(mnt) * 60 + float(sec)


def format_srt_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_captions(path, default_duration):
    entries = []  # (start, end_or_None, text)
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                head, text = line.split(None, 1)
            except ValueError:
                raise SystemExit(f"{path}:{lineno}: no caption text after the timestamp")
            if "-" in head and head.count(":") >= 2:
                start_s, end_s = head.split("-", 1)
                start, end = parse_time(start_s), parse_time(end_s)
            else:
                start, end = parse_time(head), None
            entries.append((start, end, text))

    entries.sort(key=lambda e: e[0])
    resolved = []
    for i, (start, end, text) in enumerate(entries):
        if end is None:
            next_start = entries[i + 1][0] if i + 1 < len(entries) else None
            end = start + default_duration
            if next_start is not None:
                end = min(end, next_start)
        resolved.append((start, end, text))
    return resolved


def write_srt(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(entries, 1):
            f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="input video (with audio)")
    ap.add_argument("captions", help="plain-text caption file, see format above")
    ap.add_argument("output", help="output video path")
    ap.add_argument("--duration", type=float, default=3.0,
                     help="default caption duration in seconds when no end "
                          "time is given (default 3.0)")
    ap.add_argument("--font-size", type=int, default=28)
    ap.add_argument("--margin-v", type=int, default=40,
                     help="distance from the bottom edge, in pixels")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        raise SystemExit(f"Video not found: {args.video}")
    if not os.path.isfile(args.captions):
        raise SystemExit(f"Captions file not found: {args.captions}")

    entries = parse_captions(args.captions, args.duration)
    if not entries:
        raise SystemExit("No captions found in the file.")

    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode="w",
                                      encoding="utf-8") as tmp:
        srt_path = tmp.name
    write_srt(entries, srt_path)
    print(f"{len(entries)} captions -> {srt_path}")

    # ffmpeg's subtitles filter wants a path with forward slashes and an
    # escaped colon on Windows, even inside a filter string.
    srt_for_filter = srt_path.replace("\\", "/").replace(":", "\\:")
    style = (
        f"FontName=Arial,FontSize={args.font_size},"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"BorderStyle=1,Outline=2,Shadow=0,"
        f"Alignment=2,MarginV={args.margin_v}"
    )
    vf = f"subtitles='{srt_for_filter}':force_style='{style}'"

    cmd = [
        "ffmpeg", "-y", "-i", args.video,
        "-vf", vf,
        "-c:a", "copy",
        args.output,
    ]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd)
    os.unlink(srt_path)
    if result.returncode != 0:
        raise SystemExit("ffmpeg failed -- see output above.")
    print(f"\ndone -> {args.output}")


if __name__ == "__main__":
    main()
