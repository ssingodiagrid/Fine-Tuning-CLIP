import torch
import faiss
import numpy as np
import gradio as gr
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import pickle
import os

# ------------------------------
# DEVICE
# ------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# LOAD MODEL
# ------------------------------

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

model.load_state_dict(
    torch.load("models/best_clip_model.pt", map_location=device)
)

model.to(device)
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ------------------------------
# LOAD FAISS INDEX
# ------------------------------

image_index = faiss.read_index("/Users/ssingodia/Desktop/CLIP/models/image_index.faiss")
text_index = faiss.read_index("/Users/ssingodia/Desktop/CLIP/models/text_index.faiss")

# ------------------------------
# LOAD STORED VALIDATION PAIRS
# ------------------------------

with open("/Users/ssingodia/Desktop/CLIP/data/val_pairs.pkl", "rb") as f:
    val_pairs = pickle.load(f)

# ------------------------------
# TEXT → IMAGE FUNCTION
# ------------------------------

def text_to_image(query_text):

    inputs = processor(
        text=[query_text],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)

    text_embedding = torch.nn.functional.normalize(text_embedding, dim=1)
    text_embedding = text_embedding.cpu().numpy()

    distances, indices = image_index.search(text_embedding, 5)

    results = []

    for idx in indices[0]:
        image, caption = val_pairs[idx]
        results.append(image)

    return results

# ------------------------------
# IMAGE → TEXT FUNCTION
# ------------------------------

def image_to_text(input_image):

    image = Image.fromarray(input_image).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)

    image_embedding = torch.nn.functional.normalize(image_embedding, dim=1)
    image_embedding = image_embedding.cpu().numpy()

    distances, indices = text_index.search(image_embedding, 5)

    captions = []

    for idx in indices[0]:
        _, caption = val_pairs[idx]
        captions.append(caption)

    return "\n\n".join(captions)

# ------------------------------
# GRADIO INTERFACE
# ------------------------------

with gr.Blocks() as demo:

    gr.Markdown("# 🔍 CLIP Image–Text Retrieval Demo")

    with gr.Tab("Text → Image"):

        text_input = gr.Textbox(
            label="Enter a description",
            placeholder="Example: a dog running on grass"
        )

        image_output = gr.Gallery(
            label="Top Retrieved Images",
            columns=5,
            height=300
        )

        text_button = gr.Button("Search")

        text_button.click(
            fn=text_to_image,
            inputs=text_input,
            outputs=image_output
        )

    with gr.Tab("Image → Text"):

        image_input = gr.Image(type="numpy")

        text_output = gr.Textbox(
            label="Top Retrieved Captions"
        )

        image_button = gr.Button("Search")

        image_button.click(
            fn=image_to_text,
            inputs=image_input,
            outputs=text_output
        )

demo.launch()