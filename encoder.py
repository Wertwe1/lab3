import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        B, L, H = x.size()
        
        Q = self.query(x)  # [B, L, hidden_size]
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, L, L]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)  # [B, num_heads, L, head_dim]
        context = context.transpose(1, 2).contiguous().view(B, L, self.hidden_size)  # [B, L, hidden_size]
        
        return self.dropout(self.out(context))

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        residual = x
        x = self.ln1(x)
        x = self.attn(x, mask)
        x = residual + x
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_heads=4, num_layers=4,
                 intermediate_size=512, max_len=512):
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
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        B, L = input_ids.size()
        
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(positions) + self.type_emb(token_type_ids)
        x = self.dropout(x)
        
        mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.mlm_head(x)
        print(f"Logits shape: {logits.shape}")
        return logits