# tang-poetry-llm
從零訓練的小型唐詩語言模型(字元層級 GPT,PyTorch),目的是學習 LLM 如何運作。資料來自 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) 的全唐詩。

| 檔案 | 作用 |
|---|---|
| `data.py` | 下載並清理唐詩 -> `data/tang.txt` |
| `model.py` | tokenizer + GPT 模型(含中文註解) |
| `train.py` | 訓練迴圈,定期存 checkpoint,可接續訓練 |
| `generate.py` | 給開頭,讓模型續寫 |

## 在 Colab(T4 GPU)上執行

```python
from google.colab import drive
drive.mount('/content/drive')
```
```python
!git clone https://github.com/rapxz/tang-poetry-llm.git
%cd tang-poetry-llm
!python data.py
```
```python
# 先跑 100 步確認沒問題
!python train.py --max_steps 100 --ckpt_dir /content/drive/MyDrive/tang_ckpt
```
```python
# 正式訓練(斷線後重跑同一行會自動接續)
!python train.py --max_steps 20000 --ckpt_dir /content/drive/MyDrive/tang_ckpt
```
```python
!python generate.py --prompt "春江" --ckpt_dir /content/drive/MyDrive/tang_ckpt
```

之後改程式後:`!git pull` 再訓練。
