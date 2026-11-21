# -*- coding: utf-8 -*-
"""Team data preparation: filter posts, join stock prices, label price moves (exported from a Colab notebook)."""

import pandas as pd
import os
from collections import defaultdict

# 設定關鍵字
keywords = ["群創", "洪進揚", "楊柱祥", "友達", "AUO", "華映", "CPT", "INNOLUX", "面板雙虎", "3481"]

def filter_news_by_keywords(csv_path, keywords, encoding='utf-8'):
    """
    根據關鍵字過濾新聞內容。

    參數:
    - csv_path: str，CSV 檔案的路徑
    - keywords: list[str]，關鍵字清單（可包含中英文）
    - encoding: str，CSV 檔案的編碼，預設為 utf-8

    回傳:
    - DataFrame，包含關鍵字的過濾後新聞資料
    """
    try:
        # 載入新聞 CSV
        df = pd.read_csv(csv_path, encoding=encoding)

        # 建立正則表達式 pattern
        pattern = '|'.join(keywords)

        # 過濾 content 欄位中包含任一關鍵字的新聞
        filtered_df = df[df['content'].str.contains(pattern, na=False, case=False)]

        return filtered_df

    except Exception as e:
        print(f"發生錯誤：{e}")
        return pd.DataFrame()  # 回傳空的 DataFrame

# 載入資料 並過濾
sources = ['ptt', 'mobile01','dcard_1','dcard_2','新聞']
filtered_df = []

for source in sources:
    csv_path = f"data/bda2025_202301-202503_討論數據_{source}.csv"
    df_process = filter_news_by_keywords(csv_path, keywords, encoding='utf-8')
    df_process['source'] = source
    filtered_df.append(df_process)

filtered_df = pd.concat(filtered_df, ignore_index=True)

# 每個資料來源篩選後篇數
print(filtered_df['source'].value_counts())

# 載入股價 XSLX 讀取所有工作表
sheets = pd.read_excel('data/bda2025_stock_202301-202503_上市.xlsx', sheet_name=None)

# 每個 sheet 會是一個 DataFrame，合併成一個總表
stk = pd.concat(sheets.values(), ignore_index=True)

# 將 "年月日" 欄轉成 datetime 格式
stk['年月日'] = pd.to_datetime(stk['年月日'], errors='coerce')

# 篩選代號為3481
stk_3481 = stk[(stk['代號'] == '3481')].copy()

# 按日期排序
stk_3481.sort_values('年月日', inplace=True)
stk_3481.reset_index(drop=True, inplace=True)

# 設定報酬計算天數與閾值
return_period = [1, 5, 10, 20]
return_threshold = [1,3,5]

def label_filter_news(return_period, return_threshold,stk_3481):
    # 建立 n天的報酬率（漲跌幅）
    for n in return_period:
        stk_3481[f'{n}d_return'] = stk_3481['收盤價(元)'].pct_change(periods=n) * 100

    # 建立對應的漲跌標籤
    for n in return_period:
        for r in return_threshold:
            stk_3481[f'{n}d_{r}_stock_trend'] = stk_3481[f'{n}d_return'].apply(
                lambda x: 1 if pd.notna(x) and x > r
                else (-1 if pd.notna(x) and x < -r
                else (0 if pd.notna(x) and x < r and x > -r else ''))
        )
    return stk_3481

stk_label = label_filter_news(return_period, return_threshold, stk_3481)
print(stk_label)

def attach_stock_labels_to_news(filtered_df, stk_label, return_period, return_threshold):
    """
    將股價變化（報酬與標籤）依照對應交易日，貼回到新聞資料中。

    Parameters:
    - filtered_df: 包含 post_time 的新聞 DataFrame
    - stk_label: 含標記欄位的股價資料（由 label_filter_news 傳回）
    - return_period: 如 [1, 5, 10, 20]
    - return_threshold: 如 [0.01, 0.03, 0.05]

    Returns:
    - 新聞 DataFrame，貼上對應股價資訊
    """
    # 確保日期格式正確
    filtered_df_p = filtered_df.copy()
    filtered_df_p['post_time'] = pd.to_datetime(filtered_df_p['post_time'])
    filtered_df_p['post_date'] = filtered_df_p['post_time'].dt.normalize()

    # 整理交易日清單
    trading_days = sorted(stk_label['年月日'].dropna().unique())

    # 將每個股價資料向前貼給 n 天前的交易日（新聞日）
    market_change_map = defaultdict(dict)

    for idx, row in stk_label.iterrows():
        today = row['年月日']
        if today not in trading_days:
            continue
        pos = trading_days.index(today)

        for n in return_period:
            if pos - n >= 0:
                news_day = trading_days[pos - n]
                market_change_map[news_day][f'{n}d_return'] = row.get(f'{n}d_return')
                for r in return_threshold:
                    col_name = f'{n}d_{r}_stock_trend'
                    market_change_map[news_day][col_name] = row.get(col_name)

    # 對齊最近未來的交易日
    def map_to_next_trading_day(date):
        future_days = [d for d in trading_days if d >= date]
        return future_days[0] if future_days else None

    filtered_df_p['mapped_market_day'] = filtered_df_p['post_date'].apply(map_to_next_trading_day)

    # 建立要貼的欄位名稱
    columns_to_map = [f'{n}d_return' for n in return_period]
    for n in return_period:
        for r in return_threshold:
            columns_to_map.append(f'{n}d_{r}_stock_trend')

    # 逐欄貼資料
    for col in columns_to_map:
        filtered_df_p[col] = filtered_df_p['mapped_market_day'].map(lambda d: market_change_map.get(d, {}).get(col, None))

    return filtered_df_p

# 將標記好的股價貼到新聞資料
filtered_df_labeled = attach_stock_labels_to_news(filtered_df, stk_label, return_period, return_threshold)

# 查看結果
cols_to_show = ['post_date', 'mapped_market_day', 'title'] + [f'{n}d_{r}_stock_trend' for n in return_period for r in return_threshold]
print(filtered_df_labeled[cols_to_show].head())

# 儲存
filtered_df.to_csv('data/filtered_df.csv')
stk_label.to_csv('data/stk_label.csv')
filtered_df_labeled.to_csv('data/filtered_df_labeled.csv')

# 每個return_period與return_threshold下，trend = 1 , 0 , -1 的筆數與佔比
def compute_trend_counts_and_ratios(news_df, return_period, return_threshold):
    trend_stats = []

    for n in return_period:
        for r in return_threshold:
            col = f"{n}d_{r}_stock_trend"
            if col not in news_df.columns:
                continue
            count_series = news_df[col].value_counts(dropna=True)
            total = count_series.sum()

            for trend_val in [-1, 0, 1]:
                count = count_series.get(trend_val, 0)
                ratio = count / total if total > 0 else 0
                trend_stats.append({
                    "return_period": n,
                    "threshold": r,
                    "trend": trend_val,
                    "count": count,
                    "ratio": round(ratio, 4)
                })

    return pd.DataFrame(trend_stats)

def compute_avg_returns(news_df, return_periods, thresholds, trends):
    avg_return_list = []

    for n in return_periods:
        return_col = f"{n}d_return"
        for r in thresholds:
            for t in trends:
                trend_col = f"{n}d_{r}_stock_trend"

                # 檢查欄位是否存在
                if return_col not in news_df.columns or trend_col not in news_df.columns:
                    continue

                # 篩選符合指定趨勢的樣本
                filtered_df = news_df[news_df[trend_col] == t]

                # 若無資料則跳過
                if filtered_df.empty:
                    continue

                # 計算平均報酬
                avg_return = filtered_df[return_col].mean()

                # 加入結果
                avg_return_list.append({
                    "return_period": n,
                    "threshold": r,
                    "trend": t,
                    "avg_return": round(avg_return, 4)
                })

    return pd.DataFrame(avg_return_list)

# 讀取儲存的檔案
filtered_df = pd.read_csv('data/filtered_df.csv')
stk_label = pd.read_csv('data/stk_label.csv')
filtered_df_labeled = pd.read_csv('data/filtered_df_labeled.csv')

return_period = [1, 5, 10, 20]
threshold = [1, 3, 5]
trend = [-1, 0, 1]

# 統計結果
trend_summary_df = compute_trend_counts_and_ratios(filtered_df_labeled, return_period, threshold)

# 顯示前幾筆
print("各趨勢統計（含比例）:")
print(trend_summary_df)

trends = [-1, 0, 1]

avg_return_df = compute_avg_returns(filtered_df_labeled, return_period, threshold, trend)

print("平均報酬率統計：")
print(avg_return_df)

