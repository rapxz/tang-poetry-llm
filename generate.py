"""generate.py — 給一句開頭,讓模型續寫

用法:
    python generate.py --prompt "春江花月夜" --ckpt_dir /content/drive/MyDrive/tang_ckpt
"""
import argparse
import os

import torch

from model import GPT, CharTokenizer, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="")
p.add_argument("--ckpt_dir", default="checkpoints")
p.add_argument("--max_new", type=int, default=80, help="最多生成幾個字")
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--n", type=int, default=3, help="生成幾次")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ck = torch.load(os.path.join(args.ckpt_dir, "ckpt.pt"), map_location=device)
tok = CharTokenizer(ck["chars"])  # 字表跟著 checkpoint 一起存,確保編碼一致
model = GPT(GPTConfig(**ck["cfg"])).to(device)
model.load_state_dict(ck["model"])
model.eval()
print(f"載入 step {ck['step']} 的模型")

# 訓練時詩的開頭是「詩題換行」,沒給開頭就從換行開始
ids = tok.encode(args.prompt) or tok.encode("\n")
for i in range(args.n):
    x = torch.tensor([ids], device=device)
    out = model.generate(x, args.max_new, args.temperature, args.top_k)
    print(f"--- 第 {i + 1} 首 ---\n{tok.decode(out[0].tolist())}\n")
