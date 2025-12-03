import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)     # QK^T / sqrt(d_k)
    attention_weights = softmax(scores)           # softmax
    context_vector = np.matmul(attention_weights, V)
    return attention_weights, context_vector


# ---- USER INPUT SECTION ----

print("Enter Q matrix (example: [[1,0,1]] ):")
Q = np.array(eval(input("Q = ")))

print("Enter K matrix (example: [[1,0,1],[0,1,0]] ):")
K = np.array(eval(input("K = ")))

print("Enter V matrix (example: [[5,5],[1,1]] ):")
V = np.array(eval(input("V = ")))

attn, ctx = scaled_dot_product_attention(Q, K, V)

print("\nAttention Weights:\n", attn)
print("\nContext Vector:\n", ctx)
