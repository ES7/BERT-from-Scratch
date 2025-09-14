from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from bert_model import BERTModel, BertConfig
from transformers import BertTokenizer

class DummyBERTDataset(Dataset):
    def __init__(self, tokenizer, num_samples=1000, seq_len=16, vocab_size=30522):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        token_types = torch.zeros_like(input_ids)
        attention_mask = torch.ones_like(input_ids)

        # MLM labels
        mlm_labels = input_ids.clone()
        mask = torch.rand(self.seq_len) < 0.15  # 15% tokens for MLM
        mlm_labels[~mask] = -100                # ignore non-masked tokens
        input_ids[mask] = 103                   # 103 = [MASK] token in BERT vocab

        # NSP labels
        nsp_label = torch.randint(0, 2, (1,)).item()

        return input_ids, token_types, attention_mask, mlm_labels, nsp_label

# Use HuggingFace tokenizer for real data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = DummyBERTDataset(tokenizer, num_samples=2000, seq_len=16)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Config & Model
config = BertConfig(
    vocab_size=30522,
    d_model=256,
    num_layers=2,
    n_heads=4,
    d_ff=1024,
    max_position_embeddings=128
)
model = BERTModel(config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        input_ids, token_types, attention_mask, mlm_labels, nsp_labels = batch
        optimizer.zero_grad()

        loss, mlm_logits, nsp_logits, _, _ = model(
            input_ids, token_types, attention_mask,
            mlm_labels=mlm_labels, nsp_label=nsp_labels
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")