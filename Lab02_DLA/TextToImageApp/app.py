"""
app.py

Simple text-to-image retrieval app: given a text prompt, returns the top-K
most similar images from the Flickr8k index built by build_index.py.

If no index is found on disk yet, this script builds it automatically on
first run (this only happens once -- subsequent runs load the saved index
straight away).

Run with:
    python app.py
"""

import os
import torch
import gradio as gr
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Configuration -- must be kept in sync with build_index.py
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"  # Must be the SAME checkpoint
                                                   # used in build_index.py, otherwise
                                                   # text and image embeddings would
                                                   # live in different (incompatible) spaces.
OUTPUT_DIR = os.path.join(BASE_DIR, "index")      # Where build_index.py saves its output.
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "embeddings.pt")
PATHS_PATH = os.path.join(OUTPUT_DIR, "paths.txt")
DEFAULT_TOP_K = 10   # Default number of results shown (adjustable via the slider below).
MAX_TOP_K = 50        # Upper bound offered by the slider.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Auto-build the index if it doesn't exist yet.
# This means you can just run "python app.py" directly, even the very first
# time, without having to remember to run build_index.py separately first.
# ---------------------------------------------------------------------------
if not (os.path.exists(EMBEDDINGS_PATH) and os.path.exists(PATHS_PATH)):
    print("No index found -- building it now (this happens once, may take a few minutes)...")
    from build_index import build_index
    build_index()

# ---------------------------------------------------------------------------
# Load the precomputed index ONCE, when the app starts (not on every query --
# that would mean redoing all the work build_index.py already did).
# ---------------------------------------------------------------------------
print("Loading index...")
embeddings = torch.load(EMBEDDINGS_PATH).to(device)  # [N, D]
with open(PATHS_PATH) as f:
    # One file path per line; paths[i] is the image whose embedding is at
    # embeddings[i] -- same order as when they were saved in build_index.py.
    paths = f.read().splitlines()
print(f"Loaded {len(paths)} indexed images.")

# ---------------------------------------------------------------------------
# Load CLIP ONCE at startup as well (loading it inside `search()` would
# reload the model from scratch on every single query).
# ---------------------------------------------------------------------------
print("Loading CLIP model...")
model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
model.eval()  # Inference only, no training happening here.


def search(text_query: str, top_k: int = DEFAULT_TOP_K):
    """
    Given a text prompt, return the top_k most similar images from the index.

    Returns a list of (image_path, caption) tuples, in the format Gradio's
    Gallery component expects.
    """
    # Guard against empty/whitespace-only input (e.g. before the user has
    # typed anything) -- just show nothing instead of erroring out.
    if not text_query or not text_query.strip():
        return []

    # Tokenize and encode the text query into CLIP's text-embedding space.
    # This is the same embedding space the image embeddings live in, which is
    # exactly what makes text-to-image comparison possible.
    inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    # transformers >= 5.x wraps the result in a BaseModelOutputWithPooling
    # instead of returning a plain tensor (transformers < 5.x behavior).
    # This keeps the script working with either version installed.
    if not isinstance(text_features, torch.Tensor):
        text_features = text_features.pooler_output

    # L2-normalize the text embedding too, for the same reason as in
    # build_index.py: normalized vectors turn cosine similarity into a
    # simple dot product.
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Compare the (single) text embedding against every image embedding at
    # once via matrix multiplication:
    #   embeddings: [N, D]   text_features.T: [D, 1]   -> result: [N, 1]
    # Each entry is the cosine similarity between that image and the query.
    similarities = (embeddings @ text_features.T).squeeze(1)  # -> [N]

    # Get the indices (and scores) of the top_k highest similarities.
    # min(...) guards against asking for more results than images we have.
    k = min(int(top_k), len(paths))
    top_scores, top_indices = similarities.topk(k)

    # Gradio's Gallery component accepts a list of (image, caption) tuples:
    # we use this to display the similarity score under each retrieved image.
    results = [
        (paths[idx], f"score: {score:.3f}")
        for idx, score in zip(top_indices.tolist(), top_scores.tolist())
    ]
    return results


# ---------------------------------------------------------------------------
# Gradio user interface
# ---------------------------------------------------------------------------
# gr.Blocks lets us lay out multiple components (title, textbox, button,
# slider, gallery) and wire them together explicitly, as opposed to
# gr.Interface which assumes a single input -> single output layout.
with gr.Blocks() as demo:
    gr.Markdown("## Text-to-Image Retrieval (CLIP + Flickr8k)")
    gr.Markdown("Search the Flickr8k dataset using a short text description.")

    # Textbox and button share a row, with the textbox taking up most of the
    # width (scale=9 vs scale=1) -- purely a layout/aesthetic choice.
    with gr.Row(equal_height=True):
        query = gr.Textbox(
            label="Text prompt",
            placeholder="e.g. a photo of dogs playing in the snow",
            scale=9,
        )
        search_btn = gr.Button("Retrieve", scale=1)

    # Lets the user choose how many results to see, instead of a fixed
    # hardcoded number.
    top_k_slider = gr.Slider(
        minimum=1, maximum=MAX_TOP_K, value=DEFAULT_TOP_K, step=1,
        label="Number of images",
    )

    gallery = gr.Gallery(
        label="Results",
        columns=5,
        rows=2,
        height=500,
        object_fit="cover",  # crops all images to the same size, for a tidier grid
    )

    # Two ways to trigger a search: clicking the button, or pressing Enter
    # in the textbox -- whichever the user finds more natural.
    search_btn.click(fn=search, inputs=[query, top_k_slider], outputs=gallery)
    query.submit(fn=search, inputs=[query, top_k_slider], outputs=gallery)

if __name__ == "__main__":
    # demo.launch() starts a local web server (by default at
    # http://127.0.0.1:7860) and opens the interface for interaction.
    # theme=gr.themes.Soft() just gives it a cleaner default look.
    demo.launch(theme=gr.themes.Soft())