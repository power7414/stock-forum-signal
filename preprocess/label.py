
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from tqdm import tqdm

def mark_articles_by_stock_price(articles_file, stock_price_file, output_file=None, n_days=1, threshold=0.001):
    """
    根據股價漲跌標記文章為看漲(1)、無影響(0)或看跌(-1)
    
    參數:
    articles_file: 處理過的文章 CSV 檔案路徑
    stock_price_file: 股價資料 CSV 檔案路徑
    output_file: 輸出 CSV 檔案路徑，預設為 None (自動生成)
    n_days: 預測天數，預設為 1
    threshold: 價格變化閾值，用於判定是否為無影響，預設為 0.001 (0.1%)
    
    返回:
    DataFrame: 標記後的文章資料
    """
    """
    根據股價漲跌標記文章為看漲(1)或看跌(0)
    
    參數:
    articles_file: 處理過的文章 CSV 檔案路徑
    stock_price_file: 股價資料 CSV 檔案路徑
    output_file: 輸出 CSV 檔案路徑，預設為 None (自動生成)
    
    返回:
    DataFrame: 標記後的文章資料
    """
    print(f"讀取文章資料: {articles_file}")
    articles_df = pd.read_csv(articles_file, encoding='utf-8-sig')
    
    print(f"讀取股價資料: {stock_price_file}")
    stock_df = pd.read_csv(stock_price_file)
    
    # 將日期欄位轉換為 datetime 格式
    print("處理日期格式...")
    
    # 處理股價日期格式 (可能是字串格式，需要轉換)
    if stock_df['年月日'].dtype == 'object' and '-' in str(stock_df['年月日'].iloc[0]):
        # 如果是 YYYY-MM-DD 格式
        stock_df['日期'] = pd.to_datetime(stock_df['年月日'], format='%Y-%m-%d', errors='coerce')
    else:
        # 如果是數字格式 (YYYYMMDD)
        stock_df['日期'] = pd.to_datetime(stock_df['年月日'].astype(str), format='%Y%m%d', errors='coerce')
    
    # 創建股價漲跌的字典
    print("計算股價漲跌...")
    price_change_dict = {}
    
    # 計算n天後的收盤價漲跌幅度
    stock_df['未來收盤價'] = stock_df['收盤價(元)'].shift(-n_days)
    stock_df['漲跌幅'] = (stock_df['未來收盤價'] - stock_df['收盤價(元)']) / stock_df['收盤價(元)']
    
    # 建立日期到漲跌幅的映射
    for _, row in stock_df.iterrows():
        if pd.notna(row['日期']) and pd.notna(row['漲跌幅']):
            price_change_dict[row['日期'].strftime('%Y-%m-%d')] = row['漲跌幅']
    
    # 處理文章日期格式
    print("處理文章日期...")
    
    # 創建新的欄位來存儲標記
    articles_df['article_date'] = None
    articles_df['next_trading_day'] = None
    
    # 初始化不同天數的股價變化字典
    price_change_dicts = {}
    for n in [1, 5, 10, 20]:
        # 創建新的欄位
        articles_df[f'stock_trend_{n}d'] = np.nan
        articles_df[f'price_change_{n}d'] = np.nan
        
        # 計算n天後的收盤價漲跌幅度
        stock_df[f'未來收盤價_{n}d'] = stock_df['收盤價(元)'].shift(-n)
        stock_df[f'漲跌幅_{n}d'] = (stock_df[f'未來收盤價_{n}d'] - stock_df['收盤價(元)']) / stock_df['收盤價(元)']
        
        # 初始化該天數的價格變化字典
        price_change_dicts[n] = {}
        for _, row in stock_df.iterrows():
            if pd.notna(row['日期']) and pd.notna(row[f'漲跌幅_{n}d']):
                price_change_dicts[n][row['日期'].strftime('%Y-%m-%d')] = row[f'漲跌幅_{n}d']
    
    # 處理不同來源的日期格式
    def parse_date(date_str):
        """嘗試解析不同格式的日期字串"""
        if pd.isna(date_str) or not isinstance(date_str, str):
            return None
        
        # 常見的日期格式
        formats = [
            '%Y-%m-%d %H:%M:%S',  # 2023-01-01 12:34:56
            '%Y/%m/%d %H:%M:%S',  # 2023/01/01 12:34:56
            '%Y-%m-%d',           # 2023-01-01
            '%Y/%m/%d',           # 2023/01/01
            '%Y年%m月%d日 %H:%M',  # 2023年01月01日 12:34
            '%a %b %d %H:%M:%S %Y',  # Mon Jan 01 12:34:56 2023
            '%Y-%m-%dT%H:%M:%S',  # 2023-01-01T12:34:56
            '%Y%m%d',             # 20230101
            '%Y.%m.%d %H:%M:%S',  # 2023.01.01 12:34:56
            '%Y.%m.%d',           # 2023.01.01
            '%d/%m/%Y %H:%M:%S',  # 01/01/2023 12:34:56
            '%d-%m-%Y %H:%M:%S',  # 01-01-2023 12:34:56
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        
        # 如果是 Unix 時間戳 (毫秒)
        try:
            timestamp = float(date_str) / 1000
            return datetime.fromtimestamp(timestamp).date()
        except (ValueError, TypeError):
            pass
        
        # 如果是 Unix 時間戳 (秒)
        try:
            timestamp = float(date_str)
            return datetime.fromtimestamp(timestamp).date()
        except (ValueError, TypeError):
            pass
        
        return None
    
    # 獲取股價資料的所有交易日
    trading_days = sorted([datetime.strptime(date, '%Y-%m-%d').date() for date in price_change_dict.keys()])
    
    def find_next_trading_day(article_date):
        """找到文章日期後的下一個交易日"""
        if article_date is None:
            return None
        
        # 先檢查是否有相同日期的交易日
        same_day = None
        for day in trading_days:
            if day == article_date:
                same_day = day
            elif day > article_date:
                return day
        
        # 如果找到的是同一天，也返回
        if same_day:
            return same_day
        
        return None
    
    # 標記文章
    print("標記文章...")
    
    # 處理每篇文章
    for idx, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="處理文章"):
        # 解析文章日期
        article_date = parse_date(row['post_time'])
        if article_date:
            articles_df.at[idx, 'article_date'] = article_date.strftime('%Y-%m-%d')
            
            # 找到下一個交易日
            next_trading_day = find_next_trading_day(article_date)
            
            if next_trading_day:
                next_day_str = next_trading_day.strftime('%Y-%m-%d')
                articles_df.at[idx, 'next_trading_day'] = next_day_str
                
                # 標記不同天數的股價趨勢
                for n in [1, 5, 10, 20]:
                    if next_day_str in price_change_dicts[n]:
                        price_change = price_change_dicts[n][next_day_str]
                        articles_df.at[idx, f'price_change_{n}d'] = price_change
                        
                        # 標記看漲(1)、無影響(0)或看跌(-1)
                        if abs(price_change) <= threshold:
                            articles_df.at[idx, f'stock_trend_{n}d'] = 0
                        else:
                            articles_df.at[idx, f'stock_trend_{n}d'] = 1 if price_change > 0 else -1
    
    # 檢查未標記文章
    def check_unmarked_articles(df):
        # 檢查所有天數的預測是否都被標記
        unmarked_mask = df['stock_trend_1d'].isna() | df['stock_trend_5d'].isna() | \
                       df['stock_trend_10d'].isna() | df['stock_trend_20d'].isna()
        unmarked_articles = df[unmarked_mask]
        print(f"\n未標記文章數: {len(unmarked_articles)}")
        
        if len(unmarked_articles) > 0:
            print("\n未標記文章樣本:")
            for idx, row in unmarked_articles.head(5).iterrows():
                print(f"ID: {row.get('id', 'N/A')}")
                print(f"發布時間: {row.get('post_time', 'N/A')}")
                print(f"解析後日期: {row.get('article_date', 'N/A')}")
                print(f"下一交易日: {row.get('next_trading_day', 'N/A')}")
                print("---")
    
    # 檢查未標記的文章
    check_unmarked_articles(articles_df)
    
    # 檢查未標記文章
    def check_unmarked_articles(df):
        # 檢查所有天數的預測是否都被標記
        unmarked_mask = df['stock_trend_1d'].isna() | df['stock_trend_5d'].isna() | \
                       df['stock_trend_10d'].isna() | df['stock_trend_20d'].isna()
        unmarked_articles = df[unmarked_mask]
        print(f"\n未標記文章數: {len(unmarked_articles)}")
        
        if len(unmarked_articles) > 0:
            print("\n未標記文章樣本:")
            for idx, row in unmarked_articles.head(5).iterrows():
                print(f"ID: {row.get('id', 'N/A')}")
                print(f"發布時間: {row.get('post_time', 'N/A')}")
                print(f"解析後日期: {row.get('article_date', 'N/A')}")
                print(f"下一交易日: {row.get('next_trading_day', 'N/A')}")
                print("---")
    
    # 檢查未標記的文章
    check_unmarked_articles(articles_df)
    
    # 移除無法標記的文章
    marked_articles_df = articles_df.dropna(subset=['stock_trend_1d', 'stock_trend_5d', 'stock_trend_10d', 'stock_trend_20d']).copy()
    
    # 統計結果
    total_articles = len(articles_df)
    marked_articles = len(marked_articles_df)
    
    print(f"\n===== 標記結果 =====")
    print(f"總文章數: {total_articles}")
    print(f"成功標記文章數: {marked_articles} ({marked_articles/total_articles*100:.2f}%)")
    
    for n in [1, 5, 10, 20]:
        bullish_articles = len(marked_articles_df[marked_articles_df[f'stock_trend_{n}d'] == 1])
        neutral_articles = len(marked_articles_df[marked_articles_df[f'stock_trend_{n}d'] == 0])
        bearish_articles = len(marked_articles_df[marked_articles_df[f'stock_trend_{n}d'] == -1])
        
        print(f"\n{n} 天預測結果:")
        print(f"看漲文章數 (1): {bullish_articles} ({bullish_articles/marked_articles*100:.2f}%)")
        print(f"無影響文章數 (0): {neutral_articles} ({neutral_articles/marked_articles*100:.2f}%)")
        print(f"看跌文章數 (-1): {bearish_articles} ({bearish_articles/marked_articles*100:.2f}%)")
    
    # 儲存結果
    if output_file is None:
        output_file = "marked_articles_innolux.csv"
    
    marked_articles_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n標記後的文章已儲存至: {output_file}")
    
    return marked_articles_df

# 主程式
if __name__ == "__main__":
    # 檔案路徑
    articles_file = "processed_articles.csv"
    stock_price_file = "innolux_stock_data.csv"
    output_file = "marked_articles_innolux.csv"
    
    # 執行標記
    mark_articles_by_stock_price(articles_file, stock_price_file, output_file)
