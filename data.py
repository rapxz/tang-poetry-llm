"""data.py — 下載「全唐詩」並清理成一個純文字檔 data/tang.txt

資料來源:chinese-poetry 專案(https://github.com/chinese-poetry/chinese-poetry)
的 全唐诗/poet.tang.0.json ~ poet.tang.57000.json,每個檔案約 1000 首詩。

每首詩在文字檔裡長這樣:
    詩題
    第一句。第二句。
    第三句。第四句。
    <空一行>
語言模型只是在「預測下一個字」,所以我們把所有詩接成一條超長的字串,
模型會從換行與標點的規律中,學到詩的格式。
"""
import json
import os
import urllib.parse
import urllib.request

BASE = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/"
FOLDER = "全唐诗"
# 檔案編號:0, 1000, 2000, ... 57000
FILE_IDS = range(0, 58000, 1000)

RAW_DIR = "data/raw"
OUT_FILE = "data/tang.txt"

# 含有這些字元的詩通常是殘缺、有註解或亂碼的,直接丟掉,讓資料乾淨一點
BAD_CHARS = set("□〔〕[]{}()（）《》「」『』<>#*0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def download():
    """下載所有 json(已經下載過的會跳過,所以可以重複執行)"""
    os.makedirs(RAW_DIR, exist_ok=True)
    for i in FILE_IDS:
        path = f"{RAW_DIR}/poet.tang.{i}.json"
        if os.path.exists(path):
            continue
        url = BASE + urllib.parse.quote(FOLDER) + f"/poet.tang.{i}.json"
        print("下載", url)
        urllib.request.urlretrieve(url, path)


def clean():
    """讀取所有 json,清理後寫成一個文字檔"""
    poems = []
    skipped = 0
    for i in FILE_IDS:
        with open(f"{RAW_DIR}/poet.tang.{i}.json", encoding="utf-8") as f:
            for item in json.load(f):
                title = item["title"].strip()
                lines = [p.strip() for p in item["paragraphs"] if p.strip()]
                text = title + "\n" + "\n".join(lines)
                # 太短(不成詩)或含有怪字元的就跳過
                if len(lines) < 2 or any(c in BAD_CHARS for c in text):
                    skipped += 1
                    continue
                poems.append(text)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        # 詩與詩之間空一行
        f.write("\n\n".join(poems) + "\n\n")
    print(f"保留 {len(poems)} 首,丟棄 {skipped} 首 -> {OUT_FILE}")


if __name__ == "__main__":
    download()
    clean()
    text = open(OUT_FILE, encoding="utf-8").read()
    print(f"總字數 {len(text):,},不同字元 {len(set(text)):,}")
