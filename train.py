"""train.py — 訓練迴圈

訓練的本質:
    1. 從文字中隨機切出一段(x),答案(y)就是 x 往後平移一個字
    2. 模型預測每個位置的下一個字,算出 loss
    3. 反向傳播算梯度,優化器更新參數
    4. 重複幾萬次
Colab 會斷線,所以定期存 checkpoint,重新執行同一個指令會自動接續訓練。

用法:
    python train.py --max_steps 100                       # 先跑通
    python train.py --ckpt_dir /content/drive/MyDrive/tang_ckpt --max_steps 20000
"""
import argparse
import math
import os
import time

import torch

from model import GPT, CharTokenizer, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/tang.txt")
p.add_argument("--ckpt_dir", default="checkpoints", help="checkpoint 存放資料夾(Colab 請指到 Drive)")
p.add_argument("--max_steps", type=int, default=20000)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--eval_every", type=int, default=500)
p.add_argument("--save_every", type=int, default=500)
p.add_argument("--n_layer", type=int, default=6)
p.add_argument("--n_head", type=int, default=6)
p.add_argument("--n_embd", type=int, default=384)
p.add_argument("--block_size", type=int, default=128)
args = p.parse_args()

# 自動選裝置:有 NVIDIA GPU(Colab T4)用 cuda,Mac 用 mps,否則 cpu
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1337)
os.makedirs(args.ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")

# ---------- 資料 ----------
text = open(args.data, encoding="utf-8").read()
tok = CharTokenizer.from_text(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.95 * len(data))  # 95% 訓練,5% 驗證(用來檢查有沒有過擬合)
train_data, val_data = data[:n], data[n:]
print(f"裝置 {device} | 字數 {len(data):,} | 字表 {tok.vocab_size}")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([d[i : i + args.block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + args.block_size] for i in ix])  # 答案 = 往後平移一格
    return x.to(device), y.to(device)


# ---------- 模型 ----------
cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
model = GPT(cfg).to(device)
print(f"參數量 {model.num_params() / 1e6:.2f}M")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

# T4 不支援 bfloat16,用 float16 混合精度 + GradScaler 防止梯度下溢
use_amp = device == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# ---------- 接續訓練 ----------
step = 0
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    step = ck["step"]
    print(f"從 step {step} 接續訓練")


def save():
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "cfg": cfg.__dict__, "chars": tok.chars}, ckpt_path + ".tmp")
    os.replace(ckpt_path + ".tmp", ckpt_path)  # 先寫暫存檔再改名,避免斷線時存壞


def get_lr(s):
    """學習率排程:前 warmup 步線性升高,之後用餘弦曲線降到 10%"""
    if s < args.warmup:
        return args.lr * (s + 1) / args.warmup
    prog = (s - args.warmup) / max(1, args.max_steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(20):
            x, y = get_batch(split)
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                losses.append(model(x, y)[1].item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def sample():
    ctx = torch.tensor([tok.encode("\n")], device=device)
    return tok.decode(model.generate(ctx, 80)[0].tolist())


# ---------- 訓練迴圈 ----------
model.train()
t0 = time.time()
while step < args.max_steps:
    for g in opt.param_groups:
        g["lr"] = get_lr(step)
    x, y = get_batch("train")
    with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪,防止訓練爆掉
    scaler.step(opt)
    scaler.update()
    step += 1

    if step % 50 == 0:
        print(f"step {step} | loss {loss.item():.3f} | {time.time() - t0:.0f}s")
    if step % args.eval_every == 0 or step == args.max_steps:
        l = evaluate()
        print(f"== step {step} | train {l['train']:.3f} | val {l['val']:.3f}\n{sample()}\n")
    if step % args.save_every == 0 or step == args.max_steps:
        save()
        print(f"已存 checkpoint -> {ckpt_path}")
