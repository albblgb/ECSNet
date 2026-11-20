#!/usr/bin/env bash
# Download the two public models ECSNet is built on top of.
#
#   bash tools/download_models.sh              # FLUX.1-Fill-dev + CLIP  (~35 GB)
#   bash tools/download_models.sh --clip-only  # just the CLIP used by CLIPScore
#   bash tools/download_models.sh --blip       # also BLIP, only for re-captioning
#                                              # Open Images from scratch
#
# Destination: $ECSNET_MODELS, or ./models when that is unset.
#
# FLUX.1-Fill-dev is a gated repository: accept the licence once at
# https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev and log in with
# `hf auth login` (older versions: `huggingface-cli login`) before running this.
set -euo pipefail

DEST="${ECSNET_MODELS:-./models}"
CLIP_ONLY=0
WANT_BLIP=0
for a in "$@"; do
  case "$a" in
    --clip-only) CLIP_ONLY=1 ;;
    --blip) WANT_BLIP=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $a"; exit 1 ;;
  esac
done

if command -v hf >/dev/null 2>&1; then
  DL=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  DL=(huggingface-cli download)
else
  echo "neither 'hf' nor 'huggingface-cli' found -- pip install -U huggingface_hub"
  exit 1
fi

mkdir -p "$DEST"

get () {  # repo, local directory name
  if [ -d "$DEST/$2" ] && [ -n "$(ls -A "$DEST/$2" 2>/dev/null)" ]; then
    echo "== $2 already present, skipping"
    return
  fi
  echo "== downloading $1 -> $DEST/$2"
  "${DL[@]}" "$1" --local-dir "$DEST/$2"
}

(( CLIP_ONLY )) || get black-forest-labs/FLUX.1-Fill-dev FLUX.1-Fill-dev
get openai/clip-vit-large-patch14 clip-vit-large-patch14
(( WANT_BLIP )) && get Salesforce/blip-image-captioning-large blip-image-captioning-large

echo
echo "done. Contents of $DEST:"
ls -1 "$DEST"
echo
echo "Set ECSNET_MODELS=$(cd "$DEST" && pwd) before running evaluate.py,"
echo "or leave it unset if that directory is ./models next to the code."
