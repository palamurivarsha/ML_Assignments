Name :- PALAMURI VARSHA,Id:- 700760229.
1.This project implements Linear Regression from scratch using both the Normal Equation and Gradient Descent. 
The goal is to compare both methods on synthetic data generated from the equation 𝑦 = 3 + 4 𝑥 + 𝜖 y=3+4x+ϵ.
2. This project contains 3 questions of question numbers 7,8,&9. Q7 explores Decision Trees and the effect of tree depth (underfitting vs overfitting).Q8 investigates kNN classification and how decision boundaries change with different k values.Q9 performs a full performance evaluation (confusion matrix, classification metrics, ROC, AUC) on a kNN model.

ASSIGNMENT 4:
NLP Text Processing & Named Entity Recognition

This Python assignment demonstrates basic Natural Language Processing (NLP) tasks including tokenization, stopword removal, lemmatization, POS filtering, Named Entity Recognition (NER), and pronoun ambiguity detection.

Features
Text Preprocessing:
1.Tokenization
2.Stopword removal
3.Lemmatization (not stemming)
4.Filtering to keep only verbs and nouns
Named Entity Recognition (NER):
1.Detects entities such as persons, organizations, locations, etc. using a modern NLP library.
Pronoun Ambiguity Detection:
1.Checks for pronouns like he, she, they.
2.Prints a warning if possible ambiguity is detected.
*POS filtering keeps verbs and nouns for better content analysis.
*Pronoun ambiguity detection is basic and highlights potential issues for further coreference resolution.

# Scaled Dot-Product Attention (NumPy)

This program implements the core attention mechanism used in Transformer models.  
The user provides Q, K, and V matrices, and the script computes:

- Scaled attention scores: QKᵀ / √dₖ  
- Softmax attention weights  
- Final context vector: softmax(scores) × V


---

# ✅ **README 2 — Simple Transformer Encoder Block (PyTorch)**  
**Brief + Informative**

```markdown
# Simple Transformer Encoder Block (PyTorch)

This project implements a minimal Transformer Encoder Block, focusing on the essential components:

### Components
- Multi-Head Self-Attention  
- Feed-Forward Network (Linear → ReLU → Linear)  
- **Residual Connections**  
- **Layer Normalization**  
- d_model = 128, num_heads = 8

### Input / Output
- Input shape: (32, 10, 128)  
- Output shape: (32, 10, 128)  
The encoder block preserves the original sequence shape.

### How It Works
1. `x → MultiHeadAttention → Add & LayerNorm`
2. `x → FeedForward → Add & LayerNorm`



