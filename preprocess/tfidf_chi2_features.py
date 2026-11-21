import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import os
import urllib.request

class StockTextAnalyzer:
    def __init__(self):
        # 檢查並下載詞典
        self.download_dictionary()
        
        # 載入繁體詞典
        jieba.set_dictionary('dict.txt.big')
        
        # 添加股市相關詞彙
        self.stock_terms = [
            '上漲', '下跌', '漲停', '跌停', '創高', '新低',
            '支撐', '壓力', '均線', '成交量', '籌碼',
            '營收', '獲利', '毛利', '營業額', '成長',
            '產能', '市占', '供需', '訂單', '出貨'
        ]
        for term in self.stock_terms:
            jieba.add_word(term)

    def download_dictionary(self):
        """下載繁體中文詞典"""
        dict_path = 'dict.txt.big'
        if not os.path.exists(dict_path):
            print("下載繁體中文詞典...")
            url = "https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big"
            try:
                urllib.request.urlretrieve(url, dict_path)
                print("詞典下載完成")
            except Exception as e:
                print(f"詞典下載失敗: {e}")
                print("使用預設詞典")
                return False
        return True

    def preprocess_stock_data(self, stock_df):
        """處理股價資料"""
        # 確保日期格式正確
        stock_df['date'] = pd.to_datetime(stock_df['年月日'])
        
        # 計算移動平均
        stock_df['volume_ma20'] = stock_df['成交量(千股)'].rolling(window=20).mean()
        stock_df['price_ma20'] = stock_df['收盤價(元)'].rolling(window=20).mean()
        
        return stock_df

    def calculate_significance_score(self, row):
        """計算每天的重要性分數"""
        score = 0
        
        # 價格變動分數 (最高40分)
        price_change_score = min(abs(float(row['報酬率％'])) / 2.0, 1.0) * 40
        
        # 交易量分數 (最高30分)
        volume_ratio = float(row['成交量(千股)']) / float(row['volume_ma20']) if row['volume_ma20'] > 0 else 0
        volume_score = min(volume_ratio / 1.5, 1.0) * 30
        
        # 波動度分數 (最高30分)
        volatility_score = min(float(row['高低價差%']) / 3.0, 1.0) * 30
        
        score = price_change_score + volume_score + volatility_score
        
        return score

    def mark_significant_days(self, stock_df, threshold=60):
        """標記重要交易日"""
        stock_df['significance_score'] = stock_df.apply(
            self.calculate_significance_score, axis=1
        )
        
        stock_df['is_significant'] = stock_df['significance_score'] > threshold
        
        return stock_df

    def clean_text(self, text):
        """清理文本資料"""
        if not isinstance(text, str):
            return ""
        
        # 移除特殊符號和數字
        text = re.sub(r'[^\w\s]', '', text)
        # 移除多餘空白
        text = ' '.join(text.split())
        return text

    def segment_text(self, text):
        """使用結巴進行斷詞"""
        try:
            words = jieba.lcut(text)
            # 過濾停用詞和太短的詞
            words = [w for w in words if len(w) >= 2]
            return words
        except Exception as e:
            print(f"斷詞錯誤: {e}")
            return []

    def process_articles(self, df_articles, df_stock):
        """處理文章資料"""
        print("開始處理股價資料...")
        # 預處理股價資料
        df_stock = self.preprocess_stock_data(df_stock)
        df_stock = self.mark_significant_days(df_stock)
        
        print("開始處理文章資料...")
        # 預處理文章資料
        if 'post_time' in df_articles.columns:
            df_articles['date'] = pd.to_datetime(df_articles['post_time']).dt.date
        else:
            print("警告：找不到 'post_time' 欄位，請確認資料欄位名稱")
            return None
            
        df_articles['clean_content'] = df_articles['content'].apply(self.clean_text)
        
        # 只保留重要日期的文章
        significant_dates = df_stock[df_stock['is_significant']]['date'].dt.date
        df_filtered = df_articles[df_articles['date'].isin(significant_dates)].copy()
        
        print(f"找到 {len(df_filtered)} 篇重要日期的文章")
        
        # 進行斷詞
        print("開始進行斷詞...")
        df_filtered['words'] = df_filtered['clean_content'].apply(self.segment_text)
        
        return df_filtered

    def extract_keywords(self, df_filtered, min_score=0.01):
        """提取關鍵詞"""
        if df_filtered is None or len(df_filtered) == 0:
            print("沒有找到符合條件的文章")
            return {}
            
        print("開始提取關鍵詞...")
        # 準備文檔集合
        documents = df_filtered['words'].apply(lambda x: ' '.join(x))
        
        # 計算 TF-IDF
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split())
        tfidf_matrix = tfidf.fit_transform(documents)
        
        # 獲取特徵名稱
        feature_names = tfidf.get_feature_names_out()
        
        # 計算每個詞的重要性分數
        word_scores = {}
        for word, idx in tfidf.vocabulary_.items():
            # TF-IDF 分數
            tfidf_score = np.mean(tfidf_matrix[:, idx].toarray())
            word_scores[word] = tfidf_score
        
        # 篩選分數高於閾值的關鍵詞
        significant_keywords = {
            word: score for word, score in word_scores.items() 
            if score > min_score
        }
        
        return significant_keywords

def main():
    print("開始讀取資料...")
    try:
        # 讀取資料
        df_stock = pd.read_csv('innolux_stock_data.csv')
        print("股價資料讀取成功")
        
        # 檢查必要的欄位
        required_columns = ['年月日', '成交量(千股)', '收盤價(元)', '報酬率％', '高低價差%']
        missing_columns = [col for col in required_columns if col not in df_stock.columns]
        if missing_columns:
            print(f"股價資料缺少以下欄位: {missing_columns}")
            return
            
        df_articles = pd.read_csv('processed_articles.csv')
        print("文章資料讀取成功")
        
        # 檢查必要的欄位
        if 'content' not in df_articles.columns or 'post_time' not in df_articles.columns:
            print("文章資料缺少必要欄位 (content 或 post_time)")
            return
            
    except Exception as e:
        print(f"資料讀取失敗: {e}")
        return
    
    # 初始化分析器
    analyzer = StockTextAnalyzer()
    
    # 處理文章
    df_filtered = analyzer.process_articles(df_articles, df_stock)
    
    # 提取關鍵詞
    keywords = analyzer.extract_keywords(df_filtered)
    
    if keywords:
        # 輸出結果
        result_df = pd.DataFrame({
            'keyword': keywords.keys(),
            'score': keywords.values()
        }).sort_values('score', ascending=False)
        
        print("\n關鍵詞排名：")
        print(result_df.head(20))
        
        # 保存結果
        result_df.to_csv('keywords_result.csv', index=False)
        print("\n結果已保存到 keywords_result.csv")
    else:
        print("未能產生關鍵詞")

if __name__ == "__main__":
    main()
