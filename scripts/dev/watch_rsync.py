#!/usr/bin/env python3
"""Watch a Git workspace and push changes with serialized rsync runs.

The watcher intentionally tracks only files that Git considers tracked or
untracked-and-not-ignored. This keeps generated artifacts and .gitignore'd data
from triggering syncs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_EXCLUDES = (
    ".git/",
    "**/__pycache__/**",
    "**/.ruff_cache/**",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
    "**/.cache/**",
    "**/wandb/**",
    "wandb/**",
    "**/runs/**",
    "runs/**",
    "**/logs/**",
    "**/logs_long/**",
    "**/checkpoints/**",
    "**/vis/**",
    "**/videos/**",
    "**/*checkpoint*/**",
    "**/*.mp4",
    "**/*.gif",
    "**/*.wandb",
    "**/*.pt",
    "**/*.pth",
    "**/*.safetensors",
    "**/*.parquet",
)

DEFAULT_INCLUDE_IGNORED = (
    "Self-Forcing",
    "LongLive",
)


def rsync_include_rules(path: str) -> list[str]:
    normalized = path.strip("/")
    if not normalized:
        return []
    parts = Path(normalized).parts
    rules: list[str] = []
    for index in range(1, len(parts) + 1):
        parent = Path(*parts[:index]).as_posix()
        rules.append(f"--include=/{parent}/")
    rules.append(f"--include=/{normalized}/***")
    rules.append(f"--include=/{normalized}")
    return rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Poll a Git workspace for non-ignored file changes and rsync them "
            "to a remote destination."
        )
    )
    parser.add_argument(
        "destination",
        help="Rsync destination, e.g. user@host:/remote/path/",
    )
    parser.add_argument(
        "--source",
        default=".",
        help="Local workspace to watch and sync. Defaults to current directory.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between filesystem scans. Default: 1.0.",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Wait this many quiet seconds before starting rsync. Default: 2.0.",
    )
    parser.add_argument(
        "--max-size",
        default="20m",
        help="Do not transfer files larger than this rsync size. Default: 20m.",
    )
    parser.add_argument(
        "--include-ignored",
        action="append",
        default=list(DEFAULT_INCLUDE_IGNORED),
        metavar="PATH",
        help=(
            "Also watch and sync this ignored file or directory. Can be passed "
            "multiple times. Defaults to: "
            + ", ".join(DEFAULT_INCLUDE_IGNORED)
            + "."
        ),
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help=(
            "Delete remote files that are absent locally, after filters are "
            "applied. Off by default."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rsync actions without writing remote changes.",
    )
    parser.add_argument(
        "--initial",
        action="store_true",
        help="Run one rsync immediately before watching for changes.",
    )
    parser.add_argument(
        "--rsync-path",
        default="rsync",
        help="Rsync executable to run. Default: rsync.",
    )
    return parser.parse_args()


def run_git(source: Path, args: list[str]) -> bytes:
    return subprocess.check_output(["git", *args], cwd=source)


def iter_files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return [candidate for candidate in path.rglob("*") if candidate.is_file()]


def list_watched_files(source: Path,
                       include_ignored: list[str]) -> list[Path]:
    output = run_git(source, ["ls-files", "-co", "--exclude-standard", "-z"])
    names = [name for name in output.decode("utf-8").split("\0") if name]
    paths = {Path(name) for name in names}
    for include_path in include_ignored:
        absolute_include = (source / include_path).resolve()
        if not absolute_include.is_relative_to(source):
            raise ValueError(
                f"--include-ignored path escapes source: {include_path}"
            )
        for absolute_file in iter_files_under(absolute_include):
            paths.add(absolute_file.relative_to(source))
    return sorted(paths)


def workspace_fingerprint(
    source: Path,
    include_ignored: list[str],
) -> tuple[tuple[str, int, int], ...]:
    items: list[tuple[str, int, int]] = []
    for relative_path in list_watched_files(source, include_ignored):
        absolute_path = source / relative_path
        try:
            stat = absolute_path.stat()
        except FileNotFoundError:
            items.append((relative_path.as_posix(), -1, -1))
            continue
        if absolute_path.is_file():
            items.append((relative_path.as_posix(), stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(items))


def build_rsync_command(args: argparse.Namespace, source: Path) -> list[str]:
    command = [
        args.rsync_path,
        "-azvh",
        "--progress",
        f"--max-size={args.max_size}",
    ]
    for include_path in args.include_ignored:
        command.extend(rsync_include_rules(include_path))
    command.append("--filter=:- .gitignore")
    if args.dry_run:
        command.append("--dry-run")
    if args.delete:
        command.append("--delete")
    for pattern in DEFAULT_EXCLUDES:
        command.append(f"--exclude={pattern}")
    command.extend([f"{source.as_posix().rstrip('/')}/", args.destination])
    return command


def run_rsync(args: argparse.Namespace, source: Path) -> int:
    command = build_rsync_command(args, source)
    print("$ " + " ".join(command), flush=True)
    started_at = time.monotonic()
    completed = subprocess.run(command, cwd=source)
    elapsed = time.monotonic() - started_at
    print(
        f"rsync exited with {completed.returncode} after {elapsed:.1f}s",
        flush=True,
    )
    return completed.returncode


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        print(f"source is not a directory: {source}", file=sys.stderr)
        return 2
    try:
        run_git(source, ["rev-parse", "--show-toplevel"])
    except subprocess.CalledProcessError:
        print(f"source is not inside a Git repository: {source}", file=sys.stderr)
        return 2

    print(f"Watching {source}", flush=True)
    print(f"Sync destination: {args.destination}", flush=True)
    if args.include_ignored:
        print(
            "Including ignored paths: " + ", ".join(args.include_ignored),
            flush=True,
        )
    print("Press Ctrl-C to stop.", flush=True)

    try:
        last_fingerprint = workspace_fingerprint(source, args.include_ignored)
    except subprocess.CalledProcessError as exc:
        print(f"git scan failed: {exc}", file=sys.stderr)
        return exc.returncode

    pending_since: float | None = None
    if args.initial:
        run_rsync(args, source)

    while True:
        try:
            time.sleep(args.poll_interval)
            current_fingerprint = workspace_fingerprint(
                source, args.include_ignored
            )
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
            return 0
        except subprocess.CalledProcessError as exc:
            print(f"git scan failed: {exc}; retrying", file=sys.stderr, flush=True)
            continue

        if current_fingerprint != last_fingerprint:
            last_fingerprint = current_fingerprint
            pending_since = time.monotonic()
            print("Change detected; waiting for quiet period.", flush=True)
            continue

        if pending_since is None:
            continue

        if time.monotonic() - pending_since < args.debounce:
            continue

        pending_since = None
        before_sync = last_fingerprint
        run_rsync(args, source)

        try:
            after_sync = workspace_fingerprint(source, args.include_ignored)
        except subprocess.CalledProcessError as exc:
            print(f"git scan failed: {exc}; retrying", file=sys.stderr, flush=True)
            continue

        if after_sync != before_sync:
            last_fingerprint = after_sync
            pending_since = time.monotonic()
            print("Changes occurred during rsync; queued another sync.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
