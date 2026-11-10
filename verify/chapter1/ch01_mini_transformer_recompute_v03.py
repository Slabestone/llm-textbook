"""Chapter 1 micro causal Transformer reproduction. PyTorch >=2.0; CPU only; no network.
The byte-level model is trained on a repeated-byte toy sequence; it is not a pretrained LLM.
"""
import math
import random
import torch
from torch import nn
from torch.nn import functional as F

V, D, H, L, FF, TMAX = 256, 32, 4, 2, 128, 16

def bigram_demo():
    counts = {"你": 3, "学": 2, "EOS": 1}
    p = {token: n / sum(counts.values()) for token, n in counts.items()}
    assert math.isclose(sum(p.values()), 1.0, abs_tol=1e-12)
    assert math.isclose(-math.log(p["你"]), math.log(2), abs_tol=1e-12)
    rng = random.Random(17)
    draws = rng.choices(list(p), weights=list(p.values()), k=12)
    print("BIGRAM: greedy='你', sampled=", "".join("∎" if x == "EOS" else x for x in draws))
    print(f"BIGRAM: p(你|爱)={p['你']:.3f}, NLL(你)={-math.log(p['你']):.4f}")

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3*D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.up = nn.Linear(D, FF, bias=False)
        self.down = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        packed = self.qkv(self.ln1(x)).reshape(b, t, 3, H, d//H)
        q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(b, t, d))
        return x + self.down(F.gelu(self.up(self.ln2(x))))

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(TMAX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.final_ln = nn.LayerNorm(D)

    def forward(self, ids, targets=None):
        assert ids.ndim == 2 and 1 <= ids.shape[1] <= TMAX
        b, t = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(t, device=ids.device))
        for block in self.blocks: x = block(x)
        logits = self.final_ln(x) @ self.emb.weight.T  # tied output
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: bytes, n_new: int, *, temperature=1.0,
                 sample=False, seed=123) -> bytes:
        if not prompt or len(prompt) + n_new > TMAX:
            raise ValueError("nonempty prompt and generated sequence must fit TMAX")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        ids = torch.tensor([list(prompt)], dtype=torch.long)
        rng = torch.Generator(device="cpu").manual_seed(seed)
        self.eval()
        for _ in range(n_new):
            logits, _ = self(ids)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            nxt = (torch.multinomial(probs, 1, generator=rng) if sample
                   else torch.argmax(probs, dim=-1, keepdim=True))
            ids = torch.cat([ids, nxt], dim=1)
        return bytes(ids[0].tolist())

def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    bigram_demo()
    model = TinyLM()
    count = sum(p.numel() for p in model.parameters())
    assert count == 33_600, count
    seq = torch.tensor(list(b"abababababababab"), dtype=torch.long)
    inputs, targets = seq[:8][None, :], seq[1:9][None, :]
    initial = model(inputs, targets)[1].detach().item()
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        logits, loss = model(inputs, targets)
        loss.backward()
        opt.step()
    final = model(inputs, targets)[1].detach().item()
    assert final < initial
    assert logits.shape == (1, 8, 256)
    greedy = model.generate(b"a", 7)
    sampled = model.generate(b"a", 7, temperature=0.7, sample=True, seed=123)
    assert len(greedy) == len(sampled) == 8
    assert sampled == model.generate(b"a", 7, temperature=0.7, sample=True, seed=123)
    print(f"TRAIN: {count} parameters; logits {tuple(logits.shape)}; "
          f"loss {initial:.4f} -> {final:.4f}")
    print(f"GENERATE: greedy={greedy!r}; sample T=0.7={sampled!r}")
    print("PASS: bigram probabilities, NLL, training, and reproducible decoding")

if __name__ == "__main__":
    main()
