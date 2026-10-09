# 交接文件:唐詩 GPT 訓練進度

> 給下一個 Agent / 之後的自己。最後更新:2026-10-10。

## 專案目標
從零訓練一個小型**字元層級**唐詩 GPT(PyTorch),目的是**學習 LLM 如何運作**,所以程式碼要保持精簡,並用中文註解解釋每個部分。使用者在 Mac 上用 Claude Code 寫程式,`git push` 後在 **Google Colab 免費 T4 GPU** 訓練,checkpoint 存在 Google Drive。

## 工作流程
Mac 上改程式 → `git push` → Colab `!git pull` → 訓練 → 把輸出貼回來 → 調整。
- 本機專案路徑:`/Users/kevinchen/Developer/LLM Training/tang-poetry-llm`
- GitHub:https://github.com/rapxz/tang-poetry-llm(public,帳號 `rapxz`,`gh` 已登入)
- Colab checkpoint 路徑:`/content/drive/MyDrive/tang_ckpt/ckpt.pt`
- Mac 上沒有安裝 torch;我測試時用 `/tmp/x/venv`(暫存,可能已不存在,需要時重新 `python3 -m venv` 並 `pip install torch`)

## 檔案
| 檔案 | 作用 |
|---|---|
| `data.py` | 下載 chinese-poetry `全唐诗/poet.tang.{0..57000}.json`(間隔 1000),清理後存 `data/tang.txt`。含 `□`、括號、英數字等字元的詩會被丟掉,少於 2 句的也丟 |
| `model.py` | `CharTokenizer`(字表隨 checkpoint 保存)、`GPTConfig`、`CausalSelfAttention`、`MLP`、`Block`、`GPT`(權重共享,含 `generate`,支援 temperature / top-k) |
| `train.py` | 訓練迴圈:AdamW、warmup + 餘弦學習率、梯度裁剪、T4 用 fp16 + GradScaler、每 500 步 eval 並印範例、每 500 步存 checkpoint,**重跑同一指令會自動接續** |
| `generate.py` | 載入 checkpoint,`--prompt` 續寫,可調 `--temperature --top_k --n --max_new` |

## 目前設定
- 資料:50,327 首、3,735,324 字、字表 8,862(繁體,含很多罕用字)
- 模型:6 層、6 頭、`n_embd=384`、`block_size=128`、dropout 0.1,**14.08M 參數**(比原本目標 10M 多,因為字嵌入佔約 3.4M)
- 訓練:`batch_size=64`、`lr=6e-4`、`warmup=200`、`max_steps=20000`,T4 約 28 分鐘跑完

## 訓練結果(已跑完 20000 步)
| step | train | val |
|---|---|---|
| 100 | 6.08 | 6.15 |
| 500 | 4.76 | 5.18 |
| 1000 | 4.30 | 4.84 |
| 3500 | 3.50 | **4.486(最低)** |
| 10000 | 2.73 | 4.64 |
| 20000 | 2.18 | 4.93 |

生成品質:格式正確(詩題、五七言、押韻、對仗),但 step 3500 之後 val 持續上升。

## ⚠️ 主要問題:過擬合
- val 在 ~step 3500 最低,之後一路上升,train 繼續下降。
- 原因:20000 步 × 64 × 128 ≈ 1.6 億字,資料僅 373 萬字,等於讀了約 47 個 epoch,14M 參數足以背下訓練集。
- **Drive 上現在的 `ckpt.pt` 是 step 20000(過擬合最嚴重)**。因為每次覆蓋同一個檔案,step 3500 的最佳版本已經不存在。
- 次要問題:`train.py` 的驗證集是文字檔**最後 5%**(不是隨機),原始檔案有順序性,val 分佈可能和 train 不同,數字不完全公平。

## 建議的下一步(尚未實作)
1. **按詩隨機切分**訓練/驗證集(固定 seed),而不是取尾端 5%。
2. **另存 `best.pt`**:只在 val 創新低時更新;`generate.py` 預設載入 `best.pt`。
3. **提高正則化**:dropout 0.1 → 0.2;可考慮提高 weight decay。
4. **預設步數降到約 5000**,並加 early stopping(val 連續 N 次 eval 沒進步就停)。
5. (可選)`--n_layer 4` 讓參數量更接近 10M;(可選)把低頻罕用字併成 `<unk>` 縮小字表。
6. 修改後在 Colab 刪掉舊的 `ckpt.pt` 重新訓練(舊 checkpoint 的 val 切分已不同,無法比較)。預期最佳 val 應低於 4.49。

## 注意事項
- 步數若改變,學習率排程(餘弦)是以 `--max_steps` 計算的,接續訓練時要保持一致。
- `data/`、`checkpoints/` 已在 `.gitignore`,不會上傳。
- MPS(Mac GPU)不支援注意力內的 dropout,所以 `model.py` 只保留 residual dropout;改動注意力時留意這點。
- 使用者偏好:**繁體中文回覆**、程式碼加中文註解、解釋清楚(目的是學習)。
- 對外動作(push、建 repo 等)請先跟使用者確認。
