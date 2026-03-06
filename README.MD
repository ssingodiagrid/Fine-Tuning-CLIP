# Multimodal Semantic Search Engine
## Fine-Tuned CLIP-Based Image–Text Retrieval System

This project implements a **multimodal semantic search engine** capable of performing **bidirectional retrieval between images and text** using a **fine-tuned CLIP model**.

The system improves semantic alignment between visual and textual modalities and enables **high-speed similarity search using FAISS**.

---

# Executive Summary

In today’s digital ecosystem, **over 80% of online content is visual**, but traditional keyword-based search systems struggle to understand the semantic relationship between images and text.

This project addresses this limitation by building a **multimodal retrieval system** that can:

- Retrieve **relevant images from text queries**
- Retrieve **relevant captions from images**
- Learn **shared embeddings for images and text**
- Perform **fast semantic search using FAISS**

The system fine-tunes a pretrained **CLIP (Contrastive Language–Image Pretraining)** model on the **Flickr30k dataset** using **contrastive learning**.

---

# Demo

![Demo Recording](Demo-Recording.mov)

---

# Problem Statement

Traditional search engines rely heavily on **keyword matching**, which fails to capture **semantic meaning in visual data**.

**Problem addressed:**

> Given a text query, retrieve the most semantically relevant images, and given an image query, retrieve the most relevant textual descriptions.

---

# Solution Overview

The proposed system builds a **deep learning–based multimodal retrieval engine** that:

- Learns **joint embeddings for images and captions**
- Aligns them using **contrastive learning**
- Performs **efficient similarity search using FAISS**
- Enables **real-time bidirectional retrieval**

### Workflow

```
Image / Text Input
        ↓
CLIP Encoder (Image + Text)
        ↓
512-Dimensional Embeddings
        ↓
FAISS Similarity Search
        ↓
Top-K Retrieval Results
```

---

# Dataset Description

Dataset used:

**Flickr30k (HuggingFace version: `nlphuji/flickr30k`)**

### Dataset Characteristics

- ~31,000 images
- 5 captions per image
- Natural language descriptions
- Real-world scene understanding

---

# Data Split

Since the dataset provided only a test split, a manual split was created.

| Split | Percentage | Usage |
|------|------|------|
| Train | 80% | Model training |
| Validation | 10% | Hyperparameter tuning |
| Test | 10% | Final evaluation |

---

# Data Preprocessing

The dataset was prepared as follows:

- Flattened **image–caption pairs**
- Image resizing to **224 × 224**
- Tokenization using **CLIPProcessor**
- Normalization compatible with CLIP encoders

---

# Architecture
![Architecture](/Users/ssingodia/Desktop/ds_interns_project_2_2026/Architecture.png)


## Base Model

Pretrained **CLIP (ViT-B/32)**

### Components

- **Vision Transformer (ViT)** → Image encoder  
- **Transformer Text Encoder** → Text encoder  
- **Projection layers** → Map features to a **512-dimensional shared embedding space**

Both embeddings are aligned using **contrastive learning**.

---

# Baseline Configuration

The baseline configuration establishes a **stable training pipeline** before performing architecture and hyperparameter experiments.

| Parameter | Value | Reason |
|------|------|------|
| Model | CLIP (ViT-B/32) | Widely used multimodal baseline |
| Training Strategy | Full Fine-Tuning | Allows both encoders to adapt to dataset |
| Batch Size | 32 | Balance between GPU memory and negative samples |
| Temperature | 0.07 | Stable similarity scaling used in CLIP |
| Negative Mining | In-batch negatives | Standard CLIP training strategy |
| Optimizer | AdamW | Stable optimizer for transformers |
| Learning Rate | 5e-6 | Prevents large updates in pretrained weights |


---

# Experimental Strategy

Several experiments were conducted to optimize retrieval performance.

---

# 1️⃣ Temperature Scaling

Configuration:

- Learning Rate = 5e-6  
- Batch Size = 32  
- Optimizer = AdamW  
- Negative Mining = In-batch negatives  

| Strategy | Recall@10 | Insight |
|------|------|------|
| No Scaling | 42% | Weak ranking separation |
| Fixed Temperature (0.07) | 78% | Improved semantic alignment |
| Learned Temperature | 82% | Strong ranking sharpness |

**Impact**

Learned temperature significantly improved **retrieval discrimination**.

---

# 2️⃣ Learning Rate Optimization

Configuration:

- Temperature = Learned
- Batch Size = 32
- Optimizer = AdamW
- Negative Mining = In-batch

| Vision LR | Text LR | Recall@10 | Observation |
|------|------|------|------|
| 5e-6 | 5e-6 | 82% | Higher LR causes unstable updates |
| 2e-6 | 2e-6 | 86% | Lower LR stabilizes fine-tuning |

**Conclusion**

Lower learning rates **preserve pretrained knowledge** and improve alignment.

---

# 3️⃣ Hard Negative Mining

Configuration:

- Temperature = Learned
- Batch Size = 32
- Learning Rate = 2e-6
- Optimizer = AdamW

| Strategy | Recall@10 | Impact |
|------|------|------|
| In-batch negatives | 79% | Stable baseline |
| Cross-batch negatives | 57% | Harder negatives increase training difficulty |

**Business implication**

Cross-batch negatives introduce **harder comparisons**, which can initially reduce recall but improve embedding robustness.

---

# Performance Metrics

Evaluation metrics used:

- **Recall@1**
- **Recall@5**
- **Recall@10**

---

# Final Performance

Configuration:

- Learning Rate = 2e-6
- Temperature = Learned
- Optimizer = AdamW
- Negative Mining = In-batch

| Dataset | Recall@1 | Recall@5 | Recall@10 |
|------|------|------|------|
| Validation | 54% | 79% | 86% |
| Test | 54% | 79% | 86% |

**Conclusion**

The model demonstrates **strong retrieval performance and stable generalization**, achieving **86% Recall@10** consistently across validation and test datasets.

---

# Running the Application

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.10 
*   `pip` for installing Python packages

### Installation

1.  **Clone the repository (if applicable)**
    If your project is on a version control system like Git, clone it to your local machine.
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **Create a Python virtual environment**
    It is highly recommended to use a virtual environment to manage project-specific dependencies.

    *   On macOS and Linux:
        ```bash
        python3 -m venv clipenv310
        ```

    *   On Windows:
        ```bash
        python -m venv clipenv310
        ```

3.  **Activate the virtual environment**

    *   On macOS and Linux:
        ```bash
        source clipenv310/bin/activate
        ```

    *   On Windows:
        ```bash
        .\clipenv310\Scripts\activate
        ```
    Your terminal prompt should now be prefixed with `(clipenv310)`.

4.  **Install dependencies**
    Install all the required packages from .
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the setup is complete, you can run the main application.

```bash
# Add instructions here on how to run your project's main script
# For example:
# python app.py
---

## Running the App

Once setup is complete, run the Gradio demo with:

```bash
python app.py
```

Then open the local URL printed in the terminal (e.g. `http://127.0.0.1:7860`) in your browser.

The app has two tabs:
- **Text → Image** — Enter a text description and retrieve the top 5 matching images
- **Image → Text** — Upload an image and retrieve the top 5 matching captions

---

## Required Files (Not Included in Repo)

The following files are excluded from this repository via `.gitignore` but are **required to run `app.py`**. Download links will be added below.

### Model Files (`models/`)

| File | Description | Download |
|------|-------------|----------|
| `models/best_clip_model.pt` | Fine-tuned CLIP model weights | *(https://drive.google.com/file/d/1kA85JtZgqTK6f4hqlgFpJH0zlGr-oBtQ/view?usp=sharing)* |
| `models/image_index.faiss` | FAISS index over image embeddings | *(https://drive.google.com/file/d/1wU0IqqR7O77-b9M60cKLrFRuYkXB-NHb/view?usp=drive_link)* |
| `models/text_index.faiss` | FAISS index over text embeddings | *(https://drive.google.com/file/d/15pF5pZikXIy4Y2igP61EbyUuUVN_UU1W/view?usp=sharing)* |

### Data Files (`data/`)

| File | Description | Download |
|------|-------------|----------|
| `data/val_pairs.pkl` | Validation image-caption pairs | *(https://drive.google.com/file/d/1Lli-R7QqmnbsnxMPZ2YDu2lUMQWbhOBW/view?usp=sharing)* |

> After downloading, place each file in its respective folder (`models/` or `data/`) at the root of the project before running `app.py`.# Fine_tuned_CLIP
# Fine-Tuning-CLIP
