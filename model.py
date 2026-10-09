"""model.py — 一個精簡的 GPT(字元層級)

整體流程(輸入一串字 -> 預測每個位置「下一個字」的機率):
    字 id -> 字嵌入 + 位置嵌入 -> N 個 Transformer Block -> LayerNorm -> 線性層 -> 各字的分數(logits)

每個 Block = 「自注意力(讓每個字去看前面的字)」+「前饋網路(對每個字單獨加工)」,
兩者都有殘差連接(x = x + f(x)),讓深層網路好訓練。
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 1. 字元層級 tokenizer ----------
class CharTokenizer:
    """最簡單的 tokenizer:每個不同的字元就是一個 token。

    優點:不需要任何套件,中文一個字剛好一個 token。
    缺點:序列比較長、模型要自己學「詞」的概念(現代 LLM 用 BPE 子詞切分)。
    """

    def __init__(self, chars):
        self.chars = list(chars)                           # id -> 字
        self.stoi = {c: i for i, c in enumerate(self.chars)}  # 字 -> id

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        # 不在字表裡的字直接略過
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


# ---------- 2. 模型超參數 ----------
@dataclass
class GPTConfig:
    vocab_size: int = 6000   # 字表大小(由資料決定)
    block_size: int = 128    # context 長度:模型一次最多看幾個字
    n_layer: int = 6         # Transformer Block 層數
    n_head: int = 6          # 注意力頭數
    n_embd: int = 384        # 每個字的向量維度(必須能被 n_head 整除)
    dropout: float = 0.1     # 隨機丟棄比例,防止過擬合


# ---------- 3. 因果自注意力 ----------
class CausalSelfAttention(nn.Module):
    """多頭自注意力。

    每個字產生三個向量:Query(我想找什麼)、Key(我有什麼)、Value(我的內容)。
    Q 與 K 做內積得到「相關程度」,softmax 後當權重去加權 V。
    「因果」= 每個字只能看自己和前面的字,不能偷看未來(否則預測就沒意義了)。
    多頭 = 同時用好幾組不同的 Q/K/V,各自關注不同類型的關係(對仗、押韻…)。
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)  # 一次算出 Q、K、V
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)     # 把多個頭的結果合併
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch、序列長度、向量維度
        q, k, v = self.qkv(x).split(C, dim=2)
        # 拆成多個頭: (B, T, C) -> (B, n_head, T, C/n_head)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # 等價於 softmax(QK^T / sqrt(d)) V,is_causal=True 會自動套用「不能看未來」的遮罩
        # (註:這裡不對注意力權重做 dropout,因為 Mac 的 MPS 不支援;只保留下面的 resid_drop 就夠了)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 把多個頭接回來
        return self.resid_drop(self.proj(y))


# ---------- 4. 前饋網路 ----------
class MLP(nn.Module):
    """對每個位置獨立做:放大 4 倍 -> 非線性(GELU)-> 縮回來。
    注意力負責「字與字之間交換資訊」,MLP 負責「加工每個字自己的資訊」。"""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


# ---------- 5. Transformer Block ----------
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # Pre-LN:先正規化再進子層,最後加回原本的 x(殘差連接)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------- 6. 完整的 GPT ----------
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # 字 id -> 向量
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # 位置 -> 向量(讓模型知道字的順序)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # 向量 -> 每個字的分數
        # 權重共享:輸出層與字嵌入共用同一份參數,省參數、通常也更好
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # 殘差路徑上的輸出投影用較小的初始值,避免層數多時數值膨脹
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # 共享的權重只會算一次

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # 交叉熵:模型給「正確下一個字」的機率越高,loss 越低
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """自回歸生成:每次預測下一個字,接到後面,再繼續預測。

        temperature:越小越保守(重複、安全),越大越隨機(有創意但容易亂)。
        top_k:只從機率最高的 k 個字裡抽樣,避免抽到離譜的字。
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]  # context 放不下就只留最近的字
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # 只看最後一個位置的預測
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)  # 依機率抽一個字
            idx = torch.cat([idx, nxt], dim=1)
        return idx
