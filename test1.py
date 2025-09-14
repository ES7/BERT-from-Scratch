import torch
from bert_model import BERTModel, BertConfig

config = BertConfig(
    vocab_size=30522,
    d_model=256,
    num_layers=2,
    n_heads=4,
    d_ff=1024,
    max_position_embeddings=128
)

bert = BERTModel(config)

batch_size, seq_len = 2, 16
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
token_types = torch.zeros_like(input_ids)
attention_mask = torch.ones_like(input_ids)

mlm_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
mlm_labels[0, 3] = input_ids[0, 3]
nsp_labels = torch.tensor([0, 1], dtype=torch.long)

out = bert(input_ids, token_types, attention_mask, mlm_labels=mlm_labels, nsp_label=nsp_labels)

loss, mlm_logits, nsp_logits, encoder_output, pooled_output = out

print(loss, mlm_logits.shape, nsp_logits.shape)