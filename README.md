# Personal LLM: Train Your Own Language Model from Scratch

**Author:** Nick Shin  
**Affiliation:** Fudan University

## Project Highlights

- **Complete LLM Training Pipeline**: From raw WhatsApp chat data to a fine-tuned conversational model.
- **Custom Tokenizer**: Byte Pair Encoding (BPE) implementation for efficient text processing.
- **Multiple Transformer Variants**: Explore different attention mechanisms, positional encodings, and activation functions.
- **Parameter-Efficient Fine-Tuning**: Includes LoRA (Low-Rank Adaptation) and instruction tuning.
- **RLHF**: Make a script for efficient RLHF.
- **Visualization & Analysis**: Scripts and notebooks for loss curves.

---

## Directory Structure

- `notebooks/` — Jupyter notebooks for each step: data cleaning, tokenization, model building, pre-training, fine-tuning, and advanced experiments.
- `transformer/` — Python modules for transformer architectures, attention types, and LoRA.
- `minbpe/` — BPE tokenizer implementation (adapted from Karpathy’s repo).
- `scripts/` — Utility scripts for plotting and visualization.
- `data/` — Sample WhatsApp data and templates for training/validation.
- `output/` — Generated datasets, tokenizers, and model checkpoints.
- `loss_values/` — Training logs and loss curves for various experiments.


---

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Explore the notebooks** in `notebooks/` for a step-by-step learning experience.
3. **Run scripts** in `scripts/` for visualizations and analysis.
4. **Train and fine-tune models** using code in `transformer/` and your own data in `data/`.

