# bert_from_transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.size(-1)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e9"))
        attn = scores.softmax(dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return attn @ value, attn

    def forward(self, q, k, v, mask=None):
        batch, seq, _ = q.size()
        query = self.w_q(q).view(batch, seq, self.h, self.d_k).transpose(1, 2)
        key   = self.w_k(k).view(batch, seq, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch, seq, self.h, self.d_k).transpose(1, 2)
        x, _ = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.w_o(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout)
        self.norm2 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class EncoderStack(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class BertConfig:
    def __init__(self, vocab_size=30522, d_model=768, num_layers=12, n_heads=12, d_ff=3072, max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.dropout = dropout

class BERTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layernorm = LayerNormalization(config.d_model)
        self.encoder = EncoderStack(config.num_layers, config.d_model, config.n_heads, config.d_ff, config.dropout)
        self.pooler = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.Tanh())
        self.mlm_dense = nn.Linear(config.d_model, config.d_model)
        self.mlm_act = nn.GELU()
        self.mlm_norm = LayerNormalization(config.d_model)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.nsp_classifier = nn.Linear(config.d_model, 2)

    def embed_input(self, input_ids, token_type_ids=None):
        batch_size, seq_len = input_ids.size()
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words = self.word_embeddings(input_ids)
        position_ids = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos = self.position_embeddings(position_ids)
        type_emb = self.token_type_embeddings(token_type_ids)
        x = words + pos + type_emb
        return self.dropout(self.layernorm(x))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mlm_labels=None, nsp_label=None):
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        embeddings = self.embed_input(input_ids, token_type_ids)
        encoder_output = self.encoder(embeddings, attn_mask)
        pooled_output = self.pooler(encoder_output[:, 0, :])
        mlm_hidden = self.mlm_norm(self.mlm_act(self.mlm_dense(encoder_output)))
        mlm_logits = F.linear(mlm_hidden, self.word_embeddings.weight, bias=self.mlm_bias)
        nsp_logits = self.nsp_classifier(pooled_output)

        loss = None
        if mlm_labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
        if nsp_label is not None:
            nsp_loss = nn.CrossEntropyLoss()(nsp_logits.view(-1, 2), nsp_label.view(-1))
            loss = loss + nsp_loss if loss is not None else nsp_loss
        return loss, mlm_logits, nsp_logits, encoder_output, pooled_output

if __name__ == "__main__":
    cfg = BertConfig()
    model = BERTModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    token_types = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids)
    mlm_labels = torch.full((2, 16), -100)
    mlm_labels[0, 5] = input_ids[0, 5]
    mlm_labels[1, 7] = input_ids[1, 7]
    nsp_label = torch.tensor([0, 1])
    out = model(input_ids, token_types, attention_mask, mlm_labels=mlm_labels, nsp_label=nsp_label)
    print(out[0], out[1].shape, out[2].shape)
