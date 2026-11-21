import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset 
from tqdm import tqdm
from collections import Counter
import csv


def extract_response_only(full_text):
    if "[/INST]" in full_text:
        return full_text.split("[/INST]")[-1].strip().split("</s>")[0].strip()
    return full_text.strip()

def extract_label(generated_text: str) -> int:
    text = generated_text.strip().replace("。", "").replace("．", "").replace(".", "").replace(" ", "")
    
    if "無法判斷" in text or "不確定" in text or "無法直接判斷" in text:
        return 0
    
    if text in ["1", "-1", "0"]:
        return int(text)

    if "利空" in text:
        return -1
    if "利多" in text:
        return 1

    return 0


if __name__ == "__main__":
    # === 設定 ===
    model_name = "taide/TAIDE-LX-7B-Chat"
    current_dir = Path(__file__).resolve().parent
    jsonl_path = current_dir / "train_n20.jsonl"
    pred_out_csv = "llm_predictions_300_n20_3.csv"

    # === 載入新聞資料 ===
    news = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            news.append(json.loads(line))

    # === 依時間排序後取前 300 筆 ===
    news = sorted(news, key=lambda x: x["date"])
    news = news[:300]
    print(f"🕓 資料時間範圍：{news[0]['date']} ～ {news[-1]['date']}")

    # === 轉為 Dataset ===
    dataset = Dataset.from_list(news)

    # === LLM 預測 (zero-shot 分類) ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    classifier = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20)

    # === 建立 prompt ===
    system_prompt = "請根據文字段落，判斷『群創』未來股價是利多（回答'1'）、利空（回答'-1'）、或無影響（回答'0'）。回答不需要標點符號。回答不需要文字解釋。"
    def build_prompt(example):
        question = f"「這是一則關於『群創』的社群貼文或新聞：{example['text']}。」對應的數值是什麼？'-1'？'1'？'0'？"
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{question} [/INST]"
        example["prompt"] = prompt
        return example

    print("🔍 正在產生 prompts...")
    dataset = dataset.map(build_prompt)

    print("🧠 正在逐筆使用 LLM 預測...")
    prompts = dataset["prompt"]
    results = []
    for prompt in tqdm(prompts, desc="逐筆推論"):
        full = classifier(prompt)[0]["generated_text"]
        reply = extract_response_only(full)
        results.append(reply)

    # === 合併輸出結果 ===
    dataset = dataset.add_column("generated", results)
    dataset = dataset.map(lambda x: {"label": extract_label(x["generated"])})

    # === 加入 true_label 欄位（也就是原本 json 裡的 'label'）===
    dataset = dataset.rename_column("label", "true_label")  # 先改原本的 label 名稱
    dataset = dataset.map(lambda x: {"predicted_label": extract_label(x["generated"])})  # 加入模型預測欄位

    # === 儲存結果 CSV ===
    df = dataset.to_pandas()
    df.to_csv(pred_out_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    print(f"📄 已儲存推論結果至：{pred_out_csv}")

    # === 統計模型預測結果 ===
    label_counts = Counter(df["predicted_label"])
    print("\n📊 預測 label 分布：", label_counts)

    # === 模型總準確率 ===
    correct_predictions = (df["predicted_label"] == df["true_label"]).sum()
    accuracy_all = correct_predictions / len(df) * 100

    # === 預測為利空且實際為利空的報酬率 ===
    bear_returns = df[
        (df["predicted_label"] == -1) & (df["true_label"] == -1) & (df["return"].notna())
    ]["return"].astype(float)

    # === 預測為利多且實際為利多的報酬率 ===
    bull_returns = df[
        (df["predicted_label"] == 1) & (df["true_label"] == 1) & (df["return"].notna())
    ]["return"].astype(float)

    # === 準確率（在這些條件下是否真的漲／跌）===
    accuracy_bear = (bear_returns < 0).sum() / len(bear_returns) * 100 if not bear_returns.empty else 0
    accuracy_bull = (bull_returns > 0).sum() / len(bull_returns) * 100 if not bull_returns.empty else 0

    # === 平均報酬率 ===
    avg_return_bear = bear_returns.mean() * 100 if not bear_returns.empty else 0
    avg_return_bull = bull_returns.mean() * 100 if not bull_returns.empty else 0

    # === 統一輸出格式 ===
    print(f"\n🧾 判斷為利多：{label_counts[1]} 天，無影響：{label_counts[0]} 天，利空：{label_counts[-1]} 天")
    print(f"✅ 模型總準確率：{accuracy_all:.2f}%")

    print(f"\n📉 預測為利空且實際為利空的下跌準確率：{accuracy_bear:.2f}%")
    print(f"💹 預測為利空且實際為利空的平均報酬率：{avg_return_bear:.2f}%")

    print(f"\n📈 預測為利多且實際為利多的上漲準確率：{accuracy_bull:.2f}%")
    print(f"💹 預測為利多且實際為利多的平均報酬率：{avg_return_bull:.2f}%")
