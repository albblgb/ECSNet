"""
caption_openimages.py
Generate one caption per image for the Open Images subset.

Open Images ships no captions, and the caption is the text condition the
stego branch runs on and the reference the CLIP score is measured against.
They are generated ONCE, saved, and shared by every model evaluated, so the
ECSNet-vs-FLUX comparison is not affected by caption choice.

BLIP-large is used because it was fine-tuned on COCO captions: its output is
short, literal and COCO-styled ("A man riding a horse on a beach."), which
keeps the two test sets' prompts in the same style. T5 is sensitive to prompt
style, and a captioner that writes long paragraphs would make the COCO and
Open Images numbers incomparable for a reason that has nothing to do with the
images. Name the captioner in the paper.

Usage:
    python caption_openimages.py
"""
import argparse
import json
import os

import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    _data = os.environ.get("ECSNET_DATA", "./data")
    _models = os.environ.get("ECSNET_MODELS", "./models")
    ap.add_argument("--root", default=os.path.join(_data, "openimages_val"))
    ap.add_argument("--model", default=os.path.join(_models, "blip-image-captioning-large"),
                    help="local copy of Salesforce/blip-image-captioning-large")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=30)
    ap.add_argument("--num_beams", type=int, default=3)
    a = ap.parse_args()

    from transformers import BlipForConditionalGeneration, BlipProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = BlipProcessor.from_pretrained(a.model)
    model = BlipForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    ann = json.load(open(os.path.join(a.root, "annotations.json")))["images"]
    out_imgs, out_caps = [], []
    for i in range(0, len(ann), a.batch_size):
        batch = ann[i:i + a.batch_size]
        ims = [Image.open(os.path.join(a.root, "images", x["file_name"])).convert("RGB")
               for x in batch]
        inp = proc(images=ims, return_tensors="pt").to(dev, model.dtype)
        with torch.no_grad():
            ids = model.generate(**inp, max_new_tokens=a.max_new_tokens,
                                 num_beams=a.num_beams)
        texts = proc.batch_decode(ids, skip_special_tokens=True)
        for x, t in zip(batch, texts):
            t = t.strip()
            # COCO captions are sentence-cased with a full stop; BLIP emits
            # lowercase without one. Match the style, change nothing else.
            if t:
                t = t[0].upper() + t[1:]
                if not t.endswith("."):
                    t += "."
            out_imgs.append({"id": x["id"], "file_name": x["file_name"],
                             "width": x["width"], "height": x["height"]})
            out_caps.append({"id": len(out_caps), "image_id": x["id"], "caption": t})
        print(f"  {len(out_caps)}/{len(ann)}   e.g. {texts[0]!r}")

    path = os.path.join(a.root, "captions.json")
    with open(path, "w") as f:
        json.dump({"images": out_imgs, "annotations": out_caps,
                   "captioner": os.path.basename(a.model.rstrip('/'))}, f, indent=1)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
