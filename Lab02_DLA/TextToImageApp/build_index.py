"""
build_index.py

Indexes the Flickr8k dataset using CLIP: computes an image embedding for
every image in the dataset and saves everything needed by the retrieval app
(embeddings + the images themselves, as files) to disk.

Why a separate module from app.py: computing embeddings for thousands of
images takes a few minutes, so we do it ONCE here and save the result.
app.py then just loads the saved files at startup (or calls build_index()
automatically the very first time, if no saved index is found yet).

Usage (standalone):
    python build_index.py
"""

import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME = "jxie/flickr8k"    # Hugging Face dataset identifier.
# Flickr8k on HF is split into train/validation/test (6000/1000/1000 images).
# We index ALL of them together: unlike a classification task, there is no
# training happening here, so there's no reason to hold any split back --
# using every available image just gives the retrieval app a bigger, richer
# gallery to search through.
SPLITS_TO_INDEX = ["train", "validation", "test"]

MAX_IMAGES = None                 # Set to a small number (e.g. 200) to index only a
                                   # subset -- useful for a quick first test run before
                                   # committing to indexing the full dataset.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"  # patch16 = smaller patches = finer-grained
                                                   # visual features than patch32, at the cost
                                                   # of somewhat more compute.
BATCH_SIZE = 32                   # How many images CLIP encodes at once (GPU permitting).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "index")   # Everything this script produces goes in here.
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")   # Where individual images get saved.
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "embeddings.pt")
PATHS_PATH = os.path.join(OUTPUT_DIR, "paths.txt")


def build_index():
    """
    Build the retrieval index (embeddings + saved image files) and write it
    to OUTPUT_DIR. Safe to call both as a standalone script and as an import
    from app.py (e.g. to auto-build the index the first time it's missing).
    """
    # Use the GPU if one is available (works transparently with ROCm on AMD
    # GPUs too, since PyTorch exposes the same torch.cuda.* API regardless
    # of vendor).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create the output folders if they don't already exist (does nothing if
    # you're re-running this after a previous successful/failed attempt).
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # --- Load and merge all three splits into one dataset ---
    # This downloads Flickr8k from the Hugging Face Hub the first time, then
    # reuses the local cache on subsequent runs.
    print(f"Loading dataset {DATASET_NAME} (splits={SPLITS_TO_INDEX})...")
    ds_dict = load_dataset(DATASET_NAME)
    ds = concatenate_datasets([ds_dict[split] for split in SPLITS_TO_INDEX])

    # Optionally keep only the first MAX_IMAGES examples, for a fast test run.
    if MAX_IMAGES is not None:
        ds = ds.select(range(min(MAX_IMAGES, len(ds))))
    print(f"Dataset size after merging splits: {len(ds)} examples")
    print(f"Available columns: {ds.column_names}")
    # NOTE: if the column holding the image is not called "image", check the
    # printed column names above and adjust the line "images = batch['image']"
    # further down accordingly.

    # --- Load CLIP ---
    # CLIPModel contains both the image encoder and the text encoder (we only
    # need the image side here; the text side is used later, in app.py).
    # CLIPProcessor bundles the image preprocessing (resize/crop/normalize)
    # that must exactly match what the model was trained with.
    # IMPORTANT: this must be the SAME checkpoint used in app.py, otherwise
    # text and image embeddings would live in different (incompatible) spaces.
    print(f"Loading CLIP model {CLIP_MODEL_NAME}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()  # Disable dropout etc. -- we are only doing inference, not training.

    all_embeddings = []  # Will hold one [batch, D] tensor per batch, concatenated at the end.
    all_paths = []       # Parallel list: all_paths[i] is the file path of the image
                          # whose embedding is at all_embeddings[i].

    # --- Encode images in batches ---
    # Processing images one by one would be very slow on a GPU; batching lets
    # CLIP process many images in a single forward pass.
    for start in tqdm(range(0, len(ds), BATCH_SIZE), desc="Indexing"):
        # Grab a slice of BATCH_SIZE examples from the dataset (the last
        # batch may be smaller if len(ds) isn't a multiple of BATCH_SIZE).
        batch = ds[start:start + BATCH_SIZE]
        images = batch["image"]  # A list of PIL.Image objects.

        # Preprocess the whole batch of images (resize, crop, normalize) and
        # move the resulting tensor to the GPU/CPU device.
        inputs = processor(images=images, return_tensors="pt").to(device)

        # Run the CLIP image encoder. torch.no_grad() disables gradient
        # tracking, which we don't need here (no training happening) and
        # which saves a lot of memory.
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # transformers >= 5.x wraps the result in a BaseModelOutputWithPooling
        # instead of returning a plain tensor (transformers < 5.x behavior).
        # This keeps the script working with either version installed.
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output

        # L2-normalize every embedding vector (make its length equal to 1).
        # This is what lets us later compute cosine similarity between a text
        # embedding and an image embedding as a plain dot product, instead of
        # the full cosine-similarity formula.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Move the embeddings to CPU before storing them: keeping everything
        # on the GPU would eventually run out of VRAM once we've processed
        # thousands of images.
        all_embeddings.append(image_features.cpu())

        # Save each image in this batch to disk as its own .jpg file.
        # We need this because Hugging Face datasets keep images in memory
        # (as PIL objects), not as files on disk -- but the Gradio app later
        # needs a stable file path for every image to be able to display it.
        # This also means app.py never needs to reload the dataset itself:
        # it only ever reads these local files, so it works fully offline
        # after this indexing step.
        for i, img in enumerate(images):
            idx = start + i  # Global index of this image across the whole merged dataset.
            path = os.path.join(IMAGES_DIR, f"{idx:06d}.jpg")
            img.convert("RGB").save(path)  # convert("RGB") avoids issues with
                                            # occasional grayscale/CMYK images.
            all_paths.append(path)

    # Stack every per-batch embedding tensor into one big [N, D] tensor,
    # where N = number of images and D = embedding dimension (512 for this
    # CLIP checkpoint).
    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Computed embeddings: {embeddings.shape}")

    # --- Save index to disk ---
    # embeddings.pt: the actual vectors, as a single PyTorch tensor.
    # paths.txt: one file path per line, in the same order as the embeddings,
    # so that row i of embeddings.pt corresponds to line i of paths.txt.
    torch.save(embeddings, EMBEDDINGS_PATH)
    with open(PATHS_PATH, "w") as f:
        f.write("\n".join(all_paths))

    print(f"Index saved to '{OUTPUT_DIR}/' (embeddings.pt + paths.txt + images/)")


if __name__ == "__main__":
    build_index()