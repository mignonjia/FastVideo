#!/usr/bin/env bash

set -euo pipefail

ROOT="${1:-/mnt/data/world_model/MC_data}"
OUT="${2:-/mnt/data/world_model/MC_data_extracted}"

if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required but not found in PATH" >&2
  exit 1
fi

if ! command -v unzstd >/dev/null 2>&1; then
  echo "unzstd is required but not found in PATH" >&2
  exit 1
fi

if [ ! -d "$ROOT" ]; then
  echo "Input root does not exist: $ROOT" >&2
  exit 1
fi

mkdir -p "$OUT"

extract_regular_archives() {
  find "$ROOT" -type f -name '*.tar.zst' | while read -r archive; do
    rel="${archive#$ROOT/}"
    rel_dir="$(dirname "$rel")"
    target_dir="$OUT/$rel_dir"

    mkdir -p "$target_dir"
    echo "Extracting $archive -> $target_dir"
    tar --use-compress-program=unzstd -xf "$archive" -C "$target_dir"
  done
}

extract_split_archives() {
  find "$ROOT" -type f -name '*.tar.zst.part-00' | while read -r first_part; do
    part_dir="$(dirname "$first_part")"
    base_name="$(basename "$first_part" .part-00)"
    rel_dir="${part_dir#$ROOT/}"
    target_dir="$OUT/$rel_dir"

    mkdir -p "$target_dir"
    echo "Extracting split archive $part_dir/$base_name.part-* -> $target_dir"
    cat "$part_dir/$base_name".part-* | \
      tar --use-compress-program=unzstd -xf - -C "$target_dir"
  done
}

extract_regular_archives
extract_split_archives

echo "Extraction complete: $OUT"
