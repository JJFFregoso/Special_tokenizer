import torch
import torch.nn as nn
from torch.nn import functional as F

from datetime import datetime

with open("random numbers ult.txt", "r", encoding='utf-8') as f:
    text = f.read()

batch_size = 32
block_size = 128
max_iters = 107501
eval_interval= 500
learning_rate = 5e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
eval_iters = 200
n_embd = 184
n_layer = 6
n_head = 6
dropout = 0.4


torch.manual_seed(1337)

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}



encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head self attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x= self.ln_f(x)
    logits = self.lm_head(x)
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits= logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, loss = self(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = torch.load("model_cont", weights_only=False)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# for iter in range(max_iters):
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"step {iter}: training loss {losses['train']:.4f},  val loss {losses['val']:.4f}")
#         print(datetime.now())
#     xb, yb = get_batch("train")
#
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=False)
#     loss.backward()
#     optimizer.step()

model.eval()

model_score = 0
with open("test_questions", "r", encoding="utf-8") as file:
  equations = file.read().split("\nEND")
  for i, seq in enumerate(equations):
    inputs = seq.split("ANS")
    if inputs[0] == "STOP":
      continue
    else:
      mod_input = "\n" + inputs[0]
      ans = int(inputs[1])
      context = torch.tensor((encode(mod_input), ), dtype=torch.long, device=device)
      output = decode(m.generate(context, max_new_tokens=(len(inputs[0]) + len(inputs[1]) + 3))[0].tolist())
      Z = int(output.split("= ")[1].split("\n")[0])
      if ans == 0:
          model_score += min(((ans - Z) ** 2 * 100), 50000)
      else:
          model_score += min(((((ans - Z) ** 2) / (ans ** 2)) * 100), 50000)

      print(f"epoch: {i} model score: {model_score}, ans: {ans}, mod ans: {Z}")


