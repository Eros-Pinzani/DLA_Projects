"""
app.py

Simple text-to-image retrieval app: given a text prompt, returns the top-10
most similar images from the Flickr8k index built by build_index.py.

Run with:
    python app.py
"""

import os
import torch
import gradio as gr
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
OUTPUT_DIR = "index"
TOP_K = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load the precomputed index (once, at startup) ---
print("Loading index...")
embeddings = torch.load(os.path.join(OUTPUT_DIR, "embeddings.pt")).to(device)  # [N, D]
with open(os.path.join(OUTPUT_DIR, "paths.txt")) as f:
    paths = f.read().splitlines()
print(f"Loaded {len(paths)} indexed images.")

# --- Load CLIP (once, at startup) ---
print("Loading CLIP model...")
model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
model.eval()


def search(text_query: str):
    if not text_query or not text_query.strip():
        return []

    inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        # transformers >= 5.x may wrap the result in a BaseModelOutputWithPooling
        # instead of returning a plain tensor (transformers < 5.x behavior).
        # This keeps the app working with either version installed.
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Cosine similarity = dot product, since both sides are L2-normalized.
    similarities = (embeddings @ text_features.T).squeeze(1)  # [N]

    top_scores, top_indices = similarities.topk(min(TOP_K, len(paths)))

    # Gradio's Gallery accepts a list of (image, caption) tuples: show the
    # similarity score alongside each retrieved image.
    results = [
        (paths[idx], f"score: {score:.3f}")
        for idx, score in zip(top_indices.tolist(), top_scores.tolist())
    ]
    return results


with gr.Blocks() as demo:
    gr.Markdown("## Text-to-Image Retrieval (CLIP + Flickr8k)")
    query = gr.Textbox(
        label="Text prompt",
        placeholder="e.g. a photo of dogs playing in the snow",
    )
    gallery = gr.Gallery(label="Top-10 matches", columns=5, height="auto")
    query.submit(fn=search, inputs=query, outputs=gallery)

if __name__ == "__main__":
    demo.launch()
