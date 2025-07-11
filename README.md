# Personal LLM: Train Your Own Language Model from Scratch

Welcome to the Personal LLM project! This repository is a hands-on, end-to-end guide and toolkit for training your own Large Language Model (LLM) from scratch, using real-world chat data and modern transformer architectures. The project is designed for learners, researchers, and hobbyists who want to deeply understand and experiment with every stage of the LLM pipeline.

**Author:** Nick Shin  
**Affiliation:** Fudan University

## Project Highlights

- **Complete LLM Training Pipeline**: From raw WhatsApp chat data to a fine-tuned conversational model.
- **Custom Tokenizer**: Byte Pair Encoding (BPE) implementation for efficient text processing.
- **Multiple Transformer Variants**: Explore different attention mechanisms, positional encodings, and activation functions.
- **Parameter-Efficient Fine-Tuning**: Includes LoRA (Low-Rank Adaptation) and instruction tuning.
- **Visualization & Analysis**: Scripts and notebooks for loss curves, activations, and more.
- **Extensive Documentation**: YouTube video series, slides, and detailed notebooks.

---

## Directory Structure

- `notebooks/` — Jupyter notebooks for each step: data cleaning, tokenization, model building, pre-training, fine-tuning, and advanced experiments.
- `transformer/` — Python modules for transformer architectures, attention types, and LoRA.
- `minbpe/` — BPE tokenizer implementation (adapted from Karpathy’s repo).
- `scripts/` — Utility scripts for plotting and visualization.
- `data/` — Sample WhatsApp data and templates for training/validation.
- `output/` — Generated datasets, tokenizers, and model checkpoints.
- `loss_values/` — Training logs and loss curves for various experiments.
- `colab/` — Google Colab notebook for cloud-based training.
- `images/` — Visual assets for documentation and presentations.

---

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Explore the notebooks** in `notebooks/` for a step-by-step learning experience.
3. **Run scripts** in `scripts/` for visualizations and analysis.
4. **Train and fine-tune models** using code in `transformer/` and your own data in `data/`.

---

## Learning Path

- **Data Extraction & Cleaning**: Process WhatsApp chats for NLP tasks.
- **Tokenization**: Implement and use BPE for text encoding.
- **Model Building**: Code transformer architectures from scratch.
- **Pre-training**: Train your model on your own or provided data.
- **Fine-tuning**: Instruction tuning and LoRA for conversational ability.
- **Advanced Experiments**: Try different attention mechanisms, normalization, and scaling strategies.
