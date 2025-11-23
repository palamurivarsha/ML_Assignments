import torch
import torch.nn as nn

# -----------------------
# (A) Initialize d_model=128, h=8
# -----------------------
class SimpleEncoderBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=8):
        super().__init__()

        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, d_model)
        )

        # -----------------------
        # (B) LayerNorm + Residual
        # -----------------------
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)      # Residual + Norm

        # Feed forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)       # Residual + Norm
        return x


# ----------------------------------------------------
# (C) Verify output shape → batch=32, tokens=10
# ----------------------------------------------------
if __name__ == "__main__":
    model = SimpleEncoderBlock(d_model=128, num_heads=8)

    x = torch.randn(32, 10, 128)  # (batch, seq_len, d_model)
    out = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
