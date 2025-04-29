import torch
import torch.nn as nn
import torch.nn.functional as F

# --- BERT-STYLE MODEL ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        """
        Multi‑head self‑attention.
        Args:
          hidden_size: total hidden size (will be split across heads)
          num_heads:   number of attention heads
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        """
        Args:
          x:    (batch, seq_len, hidden_size)
          mask: optional additive mask, broadcastable to (batch, num_heads, seq_len, seq_len)
        Returns:
          attn_out: (batch, seq_len, hidden_size)
        """
        bsz, seq_len, _ = x.size()

        # 1) Project to Q, K, V and split heads
        #    -> (batch, seq_len, num_heads, head_dim)
        #    -> (batch, num_heads, seq_len, head_dim)
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2) Scaled dot‑product attention
        #    scores: (batch, num_heads, seq_len, seq_len)
        scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask should be additive (zeros for keep, large neg for mask)
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)  # along seq_len of keys
        attn_out = attn_weights @ v               # (batch, num_heads, seq_len, head_dim)

        # 3) Concatenate heads and final linear
        attn_out = attn_out.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, head_dim)
        attn_out = attn_out.view(bsz, seq_len, -1)        # (batch, seq_len, hidden_size)
        attn_out = self.out_proj(attn_out)                  

        return attn_out  # (batch, seq_len, hidden_size)

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        """
        Implements the two‑layer MLP in a Transformer block:
          x → Linear(hidden_size → intermediate_size)
            → GELU()
            → Linear(intermediate_size → hidden_size)
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        x = self.fc1(x)            # → (batch, seq_len, intermediate_size)
        x = self.activation(x)     # non‑linear
        x = self.fc2(x)            # → (batch, seq_len, hidden_size)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        '''
        Args:
            x:    Tensor of shape (batch, seq_len, hidden_size)
            mask: optional attention mask of shape (batch, 1, 1, seq_len)
        '''
        # 1) Self‑attention + residual
        attn_out = self.attn(x, mask=mask)       # (batch, seq_len, hidden_size)
        x = x + attn_out
        x = self.ln1(x)

        # 2) Feed‑forward + residual
        ffn_out = self.ffn(x)                    # (batch, seq_len, hidden_size)
        x = x + ffn_out
        x = self.ln2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_heads=4, num_layers=4,
                 intermediate_size=512, max_len=512):
        '''
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
            num_layers: Number of layers
            intermediate_size: Intermediate size of the model
            max_len: Maximum length of the input
        '''
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.type_emb = nn.Embedding(2, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, input_ids, token_type_ids, attention_mask):
        batch_size, seq_len = input_ids.size()

        # Create 3 Embeddings : token, position, and token type 

        token_embeddings = self.token_emb(input_ids)  # (batch_size, seq_len, hidden_size)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.pos_emb(position_ids)
        type_embeddings = self.type_emb(token_type_ids)

        # Combine embeddings
        x = token_embeddings + position_embeddings + type_embeddings
        x = self.dropout(x)

        # Apply attention mask for bidirectional attention
        # attention_mask shape: (batch_size, seq_len) --> (batch_size, 1, 1, seq_len)
        extended_mask = attention_mask[:, None, None, :]  # broadcast over query_len
        extended_mask = (1.0 - extended_mask) * -1e9  # convert to large negative for masking

        for layer in self.layers:
            x = layer(x, extended_mask)  # expects mask with shape (batch_size, 1, 1, seq_len)

        x = self.norm(x)
        logits = self.mlm_head(x)

        return logits  # (batch_size, seq_len, vocab_size)
