# Emojify Text using LSTM & BERT

> **Self Project** | TensorFlow · Keras · HuggingFace Transformers · GloVe · NLP

---

## Overview

A 5-class emoji prediction system that maps input sentences to the most suitable emoji. Two models are implemented and compared:

1. **LSTM** — stacked LSTM network using pretrained 100-dimensional GloVe word embeddings
2. **BERT** — fine-tuned pretrained BERT Transformer for multi-class text classification

| Model | Test Accuracy | F1-Score |
|---|---|---|
| LSTM (baseline) | ~83.7% | — |
| BERT (fine-tuned) | **92.4%** | **0.91** |

BERT outperforms the LSTM baseline by **8.7 percentage points**.

---

## Emoji Classes

| Label | Emoji | Example |
|---|---|---|
| 0 | ❤️ | "I love you" |
| 1 | ⚾ | "Let's play ball" |
| 2 | 😀 | "I feel good" |
| 3 | 😞 | "I feel very bad" |
| 4 | 🍽️ | "Let's eat dinner" |

---

## Model 1 — Stacked LSTM with GloVe Embeddings

**Architecture:**
```
Embedding (GloVe 100d, frozen)
    → LSTM(16, return_sequences=True)
    → LSTM(4)
    → Dense(5, softmax)
```

**Pipeline:**
- Tokenize sentences with Keras `Tokenizer`
- Pad sequences to `maxlen` with post-padding
- One-hot encode labels via `to_categorical`
- Build embedding matrix from `glove.6B.100d.txt`
- Train with `Adam` optimizer, `categorical_crossentropy` loss, 100 epochs

---

## Model 2 — Fine-tuned BERT

**Architecture:**
```
bert-base-uncased (pretrained)
    → Dropout
    → Dense(5, softmax)
```

**Fine-tuning setup:**
- Tokenization via HuggingFace `BertTokenizer`
- Added dropout + softmax classification head
- Tuned: learning rate, batch size, number of epochs
- Evaluated with accuracy and macro F1-score

---

## Project Structure

```
emojify-text-lstm-bert/
├── Emojify_Text_LSTM.ipynb    # Full notebook (LSTM + BERT)
├── requirements.txt
└── README.md
```

### Data files needed
```
emoji_data.csv          # sentences with integer emoji labels (0–4)
glove.6B.100d.txt       # GloVe pretrained embeddings (download separately)
```

**Download GloVe embeddings:**
```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Then open `Emojify_Text_LSTM.ipynb` in Jupyter or Google Colab.

---

## Results

```
Input: "I feel good"      →  😀
Input: "I feel very bad"  →  😞
Input: "lets eat dinner"  →  🍽️
```
