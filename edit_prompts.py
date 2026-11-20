"""
edit_prompts.py
Object-replacement prompts for the editing experiment.

An instance mask covers one object of a known class. The edit replaces that
object with a different one: "a dog on the grass" -> "a tiger on the grass".

The replacement is drawn from the SAME semantic group as the source (animal
for animal, vehicle for vehicle, ...). Cross-group swaps such as car -> banana
produce prompts no user would write and make "did the edit succeed" depend on
how absurd the request is rather than on the model.

The edit prompt is the source caption with the object word substituted, when
the caption names the object; the scene description is then unchanged except
for the object. When the caption does not name it (COCO captions often say
"animal" or skip the object), the prompt falls back to "A photo of a <target>.",
and which form was used is recorded, so the two can be analysed separately.
"""
import re
from typing import Optional, Tuple

import numpy as np

GROUPS = {
    "animal": ["dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra",
               "giraffe", "tiger", "lion", "rabbit", "fox", "deer", "panda",
               "monkey", "bird", "duck", "goat", "pig"],
    "vehicle": ["car", "bus", "truck", "motorcycle", "bicycle", "train", "boat",
                "airplane", "van", "tractor"],
    "food": ["pizza", "cake", "sandwich", "apple", "banana", "orange", "broccoli",
             "donut", "hamburger", "watermelon", "strawberry", "salad"],
    "furniture": ["chair", "couch", "bed", "bench", "table", "sofa", "armchair",
                  "desk", "stool"],
    "person": ["person", "man", "woman", "boy", "girl", "child", "people",
               "human", "player", "skier", "surfer", "rider"],
    "object": ["vase", "clock", "teddy bear", "umbrella", "backpack", "suitcase",
               "bottle", "cup", "laptop", "television", "lamp", "potted plant",
               "book", "guitar", "basket", "kite", "skateboard", "surfboard"],
}
# Targets are a curated subset: things a reader immediately recognises, so the
# CLIP-based success test is measuring the edit, not the obscurity of the word.
TARGETS = {
    "animal": ["tiger", "lion", "panda", "zebra", "horse", "dog", "cat", "bear",
               "elephant", "fox"],
    "vehicle": ["car", "bus", "truck", "motorcycle", "bicycle", "boat", "tractor"],
    "food": ["pizza", "cake", "watermelon", "hamburger", "apple", "banana"],
    "furniture": ["armchair", "sofa", "bench", "desk", "stool"],
    "person": ["robot", "statue", "astronaut", "scarecrow"],
    "object": ["vase", "clock", "teddy bear", "lamp", "potted plant", "basket",
               "guitar", "backpack"],
}

_ALIASES = {"motorbike": "motorcycle", "aeroplane": "airplane", "tv": "television",
            "sofa bed": "sofa", "dining table": "table"}


def group_of(name: str) -> str:
    n = _ALIASES.get(name.lower().strip(), name.lower().strip())
    for g, words in GROUPS.items():
        if n in words:
            return g
    for g, words in GROUPS.items():     # "human face", "coffee table", ...
        if any(w in n.split() for w in words):
            return g
    return "object"


def _plural(word: str) -> str:
    head, _, last = word.rpartition(" ")
    if last.endswith(("s", "x", "ch", "sh")):
        last += "es"
    elif last.endswith("y") and last[-2:-1] not in "aeiou":
        last = last[:-1] + "ies"
    else:
        last += "s"
    return f"{head} {last}".strip()


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def make_edit(source: str, caption: str, rng: np.random.Generator
              ) -> Tuple[str, str, bool]:
    """Pick a replacement and build the edit prompt.

    Returns (target, edit_prompt, substituted). `substituted` is True when the
    source word was found in the caption and swapped in place.
    """
    src = _ALIASES.get(source.lower().strip(), source.lower().strip())
    g = group_of(src)
    pool = [t for t in TARGETS[g] if t != src and t not in src.split()]
    target = pool[int(rng.integers(0, len(pool)))]

    # Match "a dog", "dogs", "dining table" -- the original multi-word name
    # first, so "dining table" is replaced whole rather than leaving "dining".
    # The article is captured so it can be fixed ("a elephant" -> "an ..."),
    # and a plural is kept plural.
    raw = source.lower().strip()
    words = list(dict.fromkeys([raw, src] + ([src.split()[-1]] if " " in src else [])))
    for w in words:
        pat = re.compile(rf"\b(?:(a|an)\s+)?({re.escape(w)})(e?s)?\b", re.IGNORECASE)
        mt = pat.search(caption)
        if mt:
            art, plural = mt.group(1), mt.group(3)
            t = _plural(target) if plural else target
            rep = f"{_article(t)} {t}" if art else t
            if (art or mt.group(2))[0].isupper():
                rep = rep[0].upper() + rep[1:]
            return target, caption[:mt.start()] + rep + caption[mt.end():], True
    return target, f"A photo of {_article(target)} {target}.", False


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cases = [("dog", "A dog sitting on a park bench."),
             ("elephant", "Two elephants walking in the river."),
             ("car", "A red car parked next to a building."),
             ("person", "A person riding a wave on a surfboard."),
             ("human face", "A close up of a man smiling."),
             ("dining table", "A dining table with plates of food."),
             ("teddy bear", "A teddy bear on a bed."),
             ("giraffe", "Animals grazing in a field.")]
    for src, cap in cases:
        t, p, sub = make_edit(src, cap, rng)
        print(f"{src:<13} [{group_of(src):<9}] -> {t:<11} sub={sub!s:<5} {p}")
