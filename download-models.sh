#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

declare -a MODELS=(
  "hand_landmarker.task|https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

for entry in "${MODELS[@]}"; do
  name="${entry%%|*}"
  url="${entry#*|}"

  if [[ -f "$name" ]]; then
    echo "✓ $name already present, skipping"
    continue
  fi

  echo "↓ Downloading $name"
  curl --fail --location --progress-bar -o "$name.tmp" "$url"
  mv "$name.tmp" "$name"
  echo "✓ $name"
done

echo "Done."
