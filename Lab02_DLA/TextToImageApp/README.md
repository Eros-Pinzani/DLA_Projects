# TextToImageApp — Text-to-Image Retrieval (Exercise 3.3)

A simple **text-to-image retrieval** web app built with **CLIP** and
**Gradio**. It lets a user type a natural-language prompt (e.g. *"a dog
playing in the snow"*) and returns the most visually-relevant images from the
[Flickr8k](https://huggingface.co/datasets/jxie/flickr8k) dataset.

---

## How it works

The app is built around a simple **index once, query many times** pattern:

1. **Indexing (offline, done once)** — every image in Flickr8k (train +
   validation + test splits, merged into a single pool of 8,000 images) is
   passed through CLIP's image encoder to obtain a 512-dimensional,
   L2-normalized embedding. All embeddings are stacked into a single tensor
   and cached to disk, alongside a saved copy of every image — so this
   expensive step only needs to run once, and the app never needs to reach
   out to Hugging Face again afterwards.
2. **Querying (online, per request)** — when the user submits a text prompt,
   it is encoded with CLIP's text encoder into the same embedding space.
   **Cosine similarity** (a simple dot product, since all vectors are
   L2-normalized) is computed between the prompt embedding and every cached
   image embedding, and the top-K most similar images are returned and
   displayed in the UI.

This works because CLIP is trained to place matching images and text
captions close together in the same embedding space, so "similar direction"
≈ "similar meaning."

---

## Files

### `app.py`
The **Gradio user interface** and the retrieval logic. Defines a `gr.Blocks`
layout with:
- a text box for the prompt,
- a "Retrieve" button (the search also triggers on pressing Enter in the
  text box),
- a slider to choose how many images to retrieve (1–50, default 10),
- a `Gallery` component (5 columns × 2 rows, `object_fit="cover"`) showing
  the returned images with their similarity score.

On startup, it checks whether a cached index exists
(`index/embeddings.pt` + `index/paths.txt`); if not, it automatically calls
`build_index()` from `build_index.py` before starting the interface — so
running the app is always just `python app.py`, even the very first time.

### `build_index.py`
Responsible for **building the embedding index** from scratch. It:
1. Loads Flickr8k's train, validation, and test splits and merges them
   (`concatenate_datasets`) into a single pool of 8,000 images — since there
   is no training happening here, there's no reason to hold any split back.
2. Processes images in batches (`BATCH_SIZE = 32`) through CLIP's image
   encoder (`model.get_image_features`), L2-normalizing the resulting
   embeddings.
3. Saves every image as an individual `.jpg` file under `index/images/`
   (needed because Hugging Face datasets keep images in memory as PIL
   objects, not as files — but Gradio needs stable file paths to display
   them, and this also means the app has no runtime dependency on the
   `datasets` library or an internet connection once indexing is done).
4. Stacks all embeddings into a single `[8000, 512]` tensor and saves it to
   `index/embeddings.pt`, alongside `index/paths.txt` (one file path per
   line, in the same order as the embeddings).

Can be run standalone (`python build_index.py`, useful for a first test with
`MAX_IMAGES` set to a small number) or imported and called automatically by
`app.py`.

### `requirements.txt`
Python dependencies for this app: `torch`, `torchvision`, `transformers`,
`datasets`, `gradio`, `pillow`, `tqdm`.

### `index/` (generated, not checked into version control)
The cached index (`embeddings.pt`, `paths.txt`, `images/`), created
automatically the first time the app runs. Deleting this folder forces a
full re-indexing on the next launch.

---

## Running the app

From inside the `TextToImageApp/` folder:

```bash
pip install -r requirements.txt
python app.py
```

For an exact reproduction of the tested environment (pinned versions),
install from [`requirements-freeze.txt`](./requirements-freeze.txt) instead:

```bash
pip install -r requirements-freeze.txt
python app.py
```

Gradio will start a local web server (the URL is printed in the terminal) where you can enter a prompt and browse
the retrieved images.

> **First run note:** if no cached index is found, the app will first
> download Flickr8k and CLIP, then compute and save all 8,000 image
> embeddings before the interface becomes ready — this can take a few
> minutes depending on your hardware. Subsequent runs start almost instantly
> by reusing the cached index.

---

## Model

**CLIP** ([`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16))
via the Hugging Face `transformers` library. Both `build_index.py` and
`app.py` must use the exact same checkpoint, since text and image embeddings
need to live in the same space for cosine similarity to be meaningful.

> **Note on `transformers` versions:** starting with `transformers` 5.x,
> `CLIPModel.get_image_features()` / `get_text_features()` return a
> `BaseModelOutputWithPooling` object instead of a plain tensor (as in
> earlier versions). Both scripts include a small `isinstance` check so they
> work correctly with either version installed.

---

## Notes & Limitations

- Only **image embeddings** are indexed; Flickr8k's ground-truth captions
  are not currently used (e.g. for evaluation or as an additional retrieval
  signal).
- Retrieval quality is entirely determined by CLIP's pretrained embedding
  space — no fine-tuning is performed on Flickr8k for this exercise.
- Images are stored as individual files under `index/images/` rather than
  re-read from the Hugging Face dataset at query time, so the app runs fully
  offline after the index has been built once.
