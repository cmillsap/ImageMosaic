"""Generates tile_filter_prompts.npz: CLIP text embeddings for the tile filter.

The filter compares each tile's CLIP image embedding against a fixed set of
text prompts. Only the image half of CLIP ships with the app (downloaded on
demand); the text half is needed just once, here, to turn the prompts into
vectors. Re-run after editing the prompts; the .npz is committed so neither
a build nor the app needs this script.

Needs two extra packages and the text model, none of which the app uses:

    pip install tokenizers huggingface_hub
    python packaging/make_filter_prompts.py

The text model must come from the same CLIP checkpoint as
tile_filter.MODEL_URL, or the two embeddings live in different spaces.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, os.pardir, "tile_filter_prompts.npz")

REPO = "Xenova/clip-vit-base-patch32"
REVISION = "d15189d7028b43f1d3e65039190477f6af591c2a"

# Junk categories and the prompts that describe them. The category name is
# what the review list shows for a skipped tile.
JUNK = {
    "Screenshot": [
        "a screenshot of a phone screen", "a screenshot of a computer screen",
        "a screenshot of a website", "a screenshot of a text message conversation",
        "a screenshot of an app", "a screenshot of a social media post",
        "a phone lock screen with the time", "a screenshot of an Instagram story",
        "a phone screenshot with app buttons and icons",
    ],
    "Map": [
        "a map", "a screenshot of a map with directions", "a road map",
        "a satellite map view",
    ],
    "Document": [
        "a photo of a document", "a scanned page of text", "a receipt",
        "a form or paperwork", "a business card", "a page of handwritten notes",
        "a photo of a sign with text", "a whiteboard",
    ],
    "Graphic": [
        "a meme with text", "a graphic with text", "a chart or diagram",
        "a logo", "a QR code", "a cartoon or illustration", "clip art",
    ],
}

# Ordinary photos. These matter as much as the junk prompts: a softmax over
# junk prompts alone would call everything junk. The later entries were
# added after testing on a real library, where people at computers, museum
# exhibits and astrophotography were otherwise mistaken for screenshots.
PHOTO = [f"a photo of {s}" for s in (
    "people", "a person", "a child", "a baby", "a family", "a group of friends",
    "a landscape", "a city", "a street", "a building", "a room", "a house",
    "an animal", "a dog", "a cat", "food", "a meal", "flowers", "a garden",
    "a car", "the sky", "a sunset", "the beach", "the ocean", "mountains",
    "a lake", "a boat", "a party", "a sports event", "a concert", "a park",
    "trees", "snow", "an object", "a holiday celebration", "a wedding",
)] + [
    "a selfie", "a portrait", "a photo of a person sitting at a computer",
    "a photo of a person holding something", "a photo of old computers",
    "a photo of a museum exhibit", "a photo of a statue",
    "a photo of a storefront",
    "a photo of a person playing a musical instrument",
    "a photo of stars and a galaxy", "a photo of the moon",
    "a photo of electronics",
]


def main():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    def fetch(name):
        return hf_hub_download(REPO, name, revision=REVISION)

    tokenizer = Tokenizer.from_file(fetch("tokenizer.json"))
    session = ort.InferenceSession(fetch("onnx/text_model.onnx"))

    prompts = [(c, p) for c, ps in JUNK.items() for p in ps]
    prompts += [("", p) for p in PHOTO]

    vectors = []
    for _, prompt in prompts:
        ids = np.array([tokenizer.encode(prompt).ids], np.int64)
        vector = session.run(None, {"input_ids": ids})[0][0]
        vectors.append(vector / np.linalg.norm(vector))

    np.savez(OUTPUT,
             embeddings=np.stack(vectors).astype(np.float32),
             categories=np.array([c for c, _ in prompts]),
             prompts=np.array([p for _, p in prompts]))
    print(f"Wrote {len(prompts)} prompts to {os.path.normpath(OUTPUT)}")


if __name__ == "__main__":
    main()
