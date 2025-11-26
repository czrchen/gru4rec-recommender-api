import torch
import torch.nn as nn

class GRU4RecTorch(nn.Module):
    def __init__(self, n_targets, n_products, n_categories, n_brands, n_events, seq_len):
        super().__init__()

        # Embedding layers
        self.item_embed  = nn.Embedding(n_products, 256)
        self.cat_embed   = nn.Embedding(n_categories, 32)
        self.brand_embed = nn.Embedding(n_brands, 32)
        self.event_embed = nn.Embedding(n_events, 16)

        # Projection layer
        total_dim = 256 + 32 + 32 + 16       # 336
        proj_dim  = 256

        self.proj       = nn.Linear(total_dim, proj_dim)
        self.ln_proj    = nn.LayerNorm(proj_dim)
        self.drop_proj  = nn.Dropout(0.2)

        # Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        self.ln_attn = nn.LayerNorm(proj_dim)

        # GRU layers (bi-directional)
        self.gru1 = nn.GRU(proj_dim, 128, batch_first=True, bidirectional=True, dropout=0.3)
        self.gru2 = nn.GRU(256, 128, batch_first=True, bidirectional=True, dropout=0.3)

        # Dense layers
        self.bn    = nn.BatchNorm1d(256)
        self.fc1   = nn.Linear(256, 512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2   = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.4)

        # Output layer (predict category)
        self.out = nn.Linear(256, n_targets)

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, i, c, b, e):
        # Embeddings
        i = self.item_embed(i)
        c = self.cat_embed(c)
        b = self.brand_embed(b)
        e = self.event_embed(e)

        # Concatenate
        x = torch.cat([i, c, b, e], dim=-1)

        # Projection
        x = self.proj(x)
        x = self.ln_proj(x)
        x = self.gelu(x)
        x = self.drop_proj(x)

        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.ln_attn(x + attn_out)

        # GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        # Last hidden state
        x = x[:, -1, :]
        x = self.bn(x)

        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop2(x)

        return self.out(x)
