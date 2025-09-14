import torch
from transformers import BertTokenizer
from bert_model import BERTModel, BertConfig

# Load tokenizer and model config
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config = BertConfig(
    vocab_size=30522,
    d_model=256,
    num_layers=2,
    n_heads=4,
    d_ff=1024,
    max_position_embeddings=128
)
model = BERTModel(config)

# (Optional) load trained weights if saved
# model.load_state_dict(torch.load("bert_mini.pth"))

model.eval()

# Example input
sentence = "Mumbai is the [MASK] of India."
inputs = tokenizer(sentence, return_tensors="pt")

input_ids = inputs["input_ids"]
token_types = torch.zeros_like(input_ids)
attention_mask = torch.ones_like(input_ids)

# Run forward pass
with torch.no_grad():
    _, mlm_logits, _, _, _ = model(input_ids, token_types, attention_mask)

# Find predicted token for [MASK]
mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_id = mlm_logits[0, mask_index].argmax(dim=-1).item()
predicted_token = tokenizer.decode([predicted_id])

print(f"Predicted word for [MASK]: {predicted_token}")

# Testing NSP
# Sentence A and B
sentence_a = "Mumbai is the financial capital of India."
sentence_b = "It has a vibrant film industry."

# Encode
encoding = tokenizer(sentence_a, sentence_b, return_tensors="pt")
input_ids = encoding["input_ids"]
token_types = encoding["token_type_ids"]
attention_mask = encoding["attention_mask"]

# Forward pass
with torch.no_grad():
    _, _, nsp_logits, _, _ = model(input_ids, token_types, attention_mask)

# Prediction
prediction = torch.argmax(nsp_logits, dim=-1).item()
print("Prediction:", "IsNext" if prediction == 0 else "NotNext")