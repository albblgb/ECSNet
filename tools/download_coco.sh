#!/usr/bin/env bash
# download_coco.sh -- COCO 2017 in the layout ECSNet's config.py expects:
#
#   $ROOT/coco2017/
#       images/train2017/   118,287 jpg   training, and benign fine-tunes
#       images/val2017/       5,000 jpg   evaluation covers
#       annotations/        captions_*2017.json, instances_*2017.json, ...
#
#   bash download_coco.sh                  # everything (~19 GB download, ~20 GB on disk)
#   bash download_coco.sh --no-train       # val + annotations only (~1.3 GB): enough for
#                                          # every evaluation, incl. the text-message corpus
#                                          # (it uses captions_train2017.json, an annotation file)
#   bash download_coco.sh --keep-zips      # keep the archives after extraction
#
# Resumable: re-running continues partial downloads and skips finished parts.
# Each archive is integrity-checked (zip CRC) before it is extracted, and
# deleted after a successful extraction unless --keep-zips.
set -euo pipefail

ROOT="${ECSNET_DATA:-${ROOT:-./data}}"
DEST="$ROOT/coco2017"
TRAIN=1; KEEP=0
for a in "$@"; do
  case "$a" in
    --no-train) TRAIN=0 ;;
    --keep-zips) KEEP=1 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option: $a"; exit 1 ;;
  esac
done

BASE=http://images.cocodataset.org
# name | url | extract into | expected entry after extraction | expected count (0 = skip)
PARTS=("annotations_trainval2017.zip|$BASE/annotations/annotations_trainval2017.zip|$DEST|annotations/captions_val2017.json|0"
       "val2017.zip|$BASE/zips/val2017.zip|$DEST/images|val2017|5000")
(( TRAIN )) && PARTS+=("train2017.zip|$BASE/zips/train2017.zip|$DEST/images|train2017|118287")

mkdir -p "$DEST/images" "$DEST/_zips"
need_gb=$(( TRAIN ? 40 : 3 ))
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
echo "target $DEST | free ${free_gb} GB | needed during download+extract ~${need_gb} GB"
(( free_gb < need_gb )) && { echo "not enough space"; exit 1; }

if command -v aria2c >/dev/null; then
  dl() { aria2c -c -x 8 -s 8 --file-allocation=none -d "$(dirname "$2")" -o "$(basename "$2")" "$1"; }
elif command -v wget >/dev/null; then
  dl() { wget -c --tries=20 --retry-connrefused -O "$2" "$1"; }
else
  dl() { curl -L -C - --retry 20 -o "$2" "$1"; }
fi
if command -v unzip >/dev/null; then
  ztest() { unzip -tq "$1" >/dev/null; }
  zx() { unzip -q -o "$1" -d "$2"; }
else
  ztest() { python3 -c "import sys,zipfile; sys.exit(zipfile.ZipFile(sys.argv[1]).testzip() is not None)" "$1"; }
  zx() { python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"; }
fi

count_ok() {   # dir expected_count
  [[ "$2" == 0 ]] && return 0
  [[ -d "$1" ]] && (( $(find "$1" -maxdepth 1 -name '*.jpg' | wc -l) == $2 ))
}

for p in "${PARTS[@]}"; do
  IFS='|' read -r name url into entry count <<< "$p"
  target="$into/$entry"
  if [[ -e "$target" ]] && count_ok "$target" "$count"; then
    echo "[$name] already extracted -- skip"; continue
  fi
  zip="$DEST/_zips/$name"
  echo "[$name] downloading"
  dl "$url" "$zip"
  echo "[$name] checking archive"
  ztest "$zip" || { echo "[$name] corrupt archive -- delete $zip and re-run"; exit 1; }
  echo "[$name] extracting"
  zx "$zip" "$into"
  count_ok "$target" "$count" || { echo "[$name] unexpected content in $target"; exit 1; }
  (( KEEP )) || rm -f "$zip"
done
(( KEEP )) || rmdir "$DEST/_zips" 2>/dev/null || true

echo; echo "== check =="
for f in captions_train2017.json captions_val2017.json instances_train2017.json instances_val2017.json; do
  [[ -f "$DEST/annotations/$f" ]] && echo "ok   annotations/$f" || echo "MISSING annotations/$f"
done
for s in val2017 train2017; do
  [[ -d "$DEST/images/$s" ]] && echo "ok   images/$s: $(find "$DEST/images/$s" -maxdepth 1 -name '*.jpg' | wc -l) images"
done
du -sh "$DEST" | sed 's/^/size: /'

cat <<EOF

In config.py, set the root of both COCO dataset entries
(eval coco_val2017 and train_dataset) to:
  $DEST
The Open Images evaluation set is a processed subset (1000 images, masks,
BLIP captions) -- copy openimages_val/ from the old server rather than
rebuilding it.
EOF







我要取一些图片放在论文里展示，只考虑exp2的隐写任务。
我打算选择open image 的0007号图像和0029号图像作为展示
把ECSNet和比较方法的图像都选出来
最后返回一个文件夹：show_img，包括ECSNet,Balujanet, Hidden, UDH, PUSNet五个子文件夹。
每个子文件夹包含0007号图像和0029号图的所有相关图像，例如0007号图相关图像包括0007_cover.png， 0007_recovered_correct.png， 0007_residual_x10.png， 
，0007_secret_mask.png，0007_stego.png等