"""
build_index.py

Indexes the Flickr8k dataset using CLIP: computes an image embedding for every
image in the dataset and saves everything needed by the retrieval app
(embeddings + the images themselves, as files) to disk.

Run this ONCE. The resulting files (in OUTPUT_DIR) are then loaded by app.py
every time the Gradio app starts -- no need to re-index on every run.

Usage:
    python build_index.py
"""

import os
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME = "jxie/flickr8k"
SPLIT = "train"          # Flickr8k on HF is split into train/validation/test;
                          # "train" alone already has thousands of images, plenty for a demo.
MAX_IMAGES = None        # Set to a small number (e.g. 500) for a quick first test run.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
OUTPUT_DIR = "index"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # --- Load the dataset ---
    print(f"Loading dataset {DATASET_NAME} (split={SPLIT})...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    if MAX_IMAGES is not None:
        ds = ds.select(range(min(MAX_IMAGES, len(ds))))
    print(f"Dataset size: {len(ds)} examples")
    print(f"Available columns: {ds.column_names}")
    # NOTE: if the column holding the image is not called "image", check the
    # printed column names above and adjust the line "images = batch['image']"
    # further down accordingly.

    # --- Load CLIP ---
    print(f"Loading CLIP model {CLIP_MODEL_NAME}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    all_embeddings = []
    all_paths = []

    # --- Encode images in batches ---
    for start in tqdm(range(0, len(ds), BATCH_SIZE), desc="Indexing"):
        batch = ds[start:start + BATCH_SIZE]

        images = batch["image"]

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # transformers >= 5.x wraps the result in a BaseModelOutputWithPooling
        # instead of returning a plain tensor (transformers < 5.x behavior).
        # This keeps the script working with either version installed.
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        # L2-normalize so that cosine similarity becomes a simple dot product.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        all_embeddings.append(image_features.cpu())

        # Save each image to disk as a stable file the Gradio app can display,
        # since the in-memory HF dataset images don't have permanent file paths.
        for i, img in enumerate(images):
            idx = start + i
            path = os.path.join(IMAGES_DIR, f"{idx:06d}.jpg")
            img.convert("RGB").save(path)
            all_paths.append(path)

    embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
    print(f"Computed embeddings: {embeddings.shape}")

    # --- Save index to disk ---
    torch.save(embeddings, os.path.join(OUTPUT_DIR, "embeddings.pt"))
    with open(os.path.join(OUTPUT_DIR, "paths.txt"), "w") as f:
        f.write("\n".join(all_paths))

    print(f"Index saved to '{OUTPUT_DIR}/' (embeddings.pt + paths.txt + images/)")


if __name__ == "__main__":
    main()