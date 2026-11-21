import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from scipy.sparse import csr_matrix, save_npz
import json
from tqdm import tqdm
from collections import Counter
import os
import re
import math
import time

# 設定 jieba 分詞
def setup_jieba():
    """設定 jieba 分詞，包括載入停用詞"""
    print("設定 jieba 分詞...")
    
    # 設定停用詞
    stopwords = set([
        "的", "了", "和", "是", "就", "都", "而", "及", "與", "著",
        "或", "一個", "沒有", "因為", "但是", "然而", "如果", "一些",
        "很多", "這個", "那個", "這些", "那些", "進行", "不是", "這樣",
        "一樣", "所以", "只是", "大家", "出來", "我們", "他們", "你們",
        "可以", "不能", "這", "那", "之", "得", "也", "要", "會", "對"
    ])
    
    return stopwords

def preprocess_text(text):
    """預處理文本"""
    if not isinstance(text, str):
        return ""
    
    # 移除非中文字符（保留英文字母、數字和一些標點符號）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,，。、：:；;！!？?（）()%％]', ' ', text)
    
    # 移除多餘空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def segment_text(text, stopwords):
    """對文本進行分詞並過濾停用詞"""
    if not isinstance(text, str) or not text:
        return []
    
    # 進行分詞
    words = jieba.cut(text)
    
    # 過濾停用詞和單字詞
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
    
    return filtered_words

def calculate_tf_idf(article_words_list, min_df=5):
    """計算 TF-IDF 值"""
    print("計算 TF-IDF 值...")
    
    # 統計詞頻 (TF)
    all_words = []
    for words in tqdm(article_words_list, desc="統計詞頻"):
        all_words.extend(words)
    
    word_tf = Counter(all_words)
    
    # 統計文檔頻率 (DF)
    word_df = {}
    for words in tqdm(article_words_list, desc="統計文檔頻率"):
        for word in set(words):  # 使用 set 確保每篇文章中每個詞只計算一次
            word_df[word] = word_df.get(word, 0) + 1
    
    # 過濾低頻詞
    filtered_words = {word: df for word, df in word_df.items() if df >= min_df}
    print(f"過濾後的詞彙數: {len(filtered_words)}")
    
    # 計算 TF-IDF
    n_docs = len(article_words_list)
    word_tfidf = {}
    
    for word, df in tqdm(filtered_words.items(), desc="計算 TF-IDF"):
        tf = word_tf[word]
        idf = math.log(n_docs / (df + 1)) + 1  # 添加平滑處理，避免除零
        word_tfidf[word] = tf * idf
    
    return word_tfidf, word_tf, word_df, n_docs

def calculate_chi_square(article_words_list, labels, min_df=5):
    """計算卡方統計值"""
    print("計算卡方統計值...")
    
    # 統計每個類別的文檔數
    class_counts = Counter(labels)
    n_docs = len(article_words_list)
    
    # 統計每個詞在每個類別中的出現次數
    word_class_counts = {}
    word_df = {}
    
    for i, words in tqdm(enumerate(article_words_list), desc="統計詞彙類別分布", total=len(article_words_list)):
        label = labels[i]
        for word in set(words):  # 使用 set 確保每篇文章中每個詞只計算一次
            if word not in word_class_counts:
                word_class_counts[word] = Counter()
            word_class_counts[word][label] += 1
            word_df[word] = word_df.get(word, 0) + 1
    
    # 過濾低頻詞
    filtered_words = {word: df for word, df in word_df.items() if df >= min_df}
    print(f"過濾後的詞彙數: {len(filtered_words)}")
    
    # 計算卡方值
    chi_squares = {}
    
    for word in tqdm(filtered_words, desc="計算卡方值"):
        chi_square = 0
        word_counts = word_class_counts[word]
        word_total = sum(word_counts.values())
        
        for label, class_count in class_counts.items():
            # 計算觀察值
            observed = word_counts.get(label, 0)
            
            # 計算期望值
            expected = (word_total * class_count) / n_docs
            
            # 計算卡方貢獻
            if expected > 0:
                chi_square += ((observed - expected) ** 2) / expected
        
        chi_squares[word] = chi_square
    
    return chi_squares

def calculate_continuous_price_changes(articles_df):
    """計算連續範圍的股價變動"""
    print("計算連續範圍的股價變動...")
    
    # 確保 stock_trend_1d 和 date 存在
    if 'stock_trend_1d' not in articles_df.columns:
        print("錯誤：找不到 stock_trend_1d 欄位")
        return articles_df
    
    if 'date' not in articles_df.columns:
        print("錯誤：找不到 date 欄位，無法計算連續範圍")
        return articles_df
    
    # 確保 date 欄位是日期類型
    articles_df['date'] = pd.to_datetime(articles_df['date'])
    
    # 按日期排序
    articles_df = articles_df.sort_values('date')
    
    # 計算 5d 連續範圍 (d~d+4)
    articles_df['stock_trend_5d_continuous'] = 0
    
    # 為每篇文章，查找接下來4天的文章並計算平均股價變動
    for i in tqdm(range(len(articles_df)), desc="計算 5d 連續範圍"):
        # 獲取當前文章的日期
        current_date = articles_df.iloc[i]['date']
        
        # 找出從當天開始的5天內的所有文章
        next_4_days = articles_df[(articles_df['date'] >= current_date) & 
                                 (articles_df['date'] <= pd.to_datetime(current_date) + pd.Timedelta(days=4))]
        
        # 計算這5天內的平均股價變動
        if len(next_4_days) > 0:
            avg_change = next_4_days['stock_trend_1d'].mean()
            articles_df.loc[articles_df.index[i], 'stock_trend_5d_continuous'] = avg_change
    
    # 10d (d~d+9)
    articles_df['stock_trend_10d_continuous'] = 0
    for i in tqdm(range(len(articles_df)), desc="計算 10d 連續範圍"):
        current_date = articles_df.iloc[i]['date']
        next_9_days = articles_df[(articles_df['date'] >= current_date) & 
                                 (articles_df['date'] <= pd.to_datetime(current_date) + pd.Timedelta(days=9))]
        if len(next_9_days) > 0:
            avg_change = next_9_days['stock_trend_1d'].mean()
            articles_df.loc[articles_df.index[i], 'stock_trend_10d_continuous'] = avg_change
    
    # 20d (d~d+19)
    articles_df['stock_trend_20d_continuous'] = 0
    for i in tqdm(range(len(articles_df)), desc="計算 20d 連續範圍"):
        current_date = articles_df.iloc[i]['date']
        next_19_days = articles_df[(articles_df['date'] >= current_date) & 
                                  (articles_df['date'] <= pd.to_datetime(current_date) + pd.Timedelta(days=19))]
        if len(next_19_days) > 0:
            avg_change = next_19_days['stock_trend_1d'].mean()
            articles_df.loc[articles_df.index[i], 'stock_trend_20d_continuous'] = avg_change
    
    return articles_df

def analyze_articles(csv_file, min_df=5):
    """分析文章，提取詞彙特徵"""
    start_time = time.time()
    print(f"分析文章文件: {csv_file}")
    
    # 設定 jieba 分詞
    stopwords = setup_jieba()
    
    # 讀取 CSV 文件
    print("讀取 CSV 文件...")
    df = pd.read_csv(csv_file)
    print(f"共讀取 {len(df)} 篇文章")
    
    # 計算連續範圍的股價變動
    df = calculate_continuous_price_changes(df)
    
    # 預處理和分詞
    print("進行文本預處理和分詞...")
    article_words_list = []
    
    for content in tqdm(df['content'].fillna(''), desc="處理文章"):
        # 預處理文本
        preprocessed_text = preprocess_text(content)
        
        # 分詞
        words = segment_text(preprocessed_text, stopwords)
        
        article_words_list.append(words)
    
    # 計算詞彙統計
    print("計算詞彙統計...")
    
    # 計算 TF-IDF
    word_tfidf, word_tf, word_df, n_docs = calculate_tf_idf(article_words_list, min_df)
    
    # 計算卡方統計值 (使用 1d 股價趨勢作為標籤)
    # 將股價趨勢轉換為類別標籤 (1: 上漲, 0: 不變, -1: 下跌)
    labels = []
    for trend in df['stock_trend_1d'].fillna(0):
        if trend > 0.01:  # 上漲超過1%
            labels.append(1)
        elif trend < -0.01:  # 下跌超過1%
            labels.append(-1)
        else:
            labels.append(0)
    
    chi_squares = calculate_chi_square(article_words_list, labels, min_df)
    
    # 計算綜合分數 (TF-IDF * sqrt(卡方值))
    word_scores = {}
    for word in tqdm(word_tfidf.keys(), desc="計算綜合分數"):
        if word in chi_squares:
            word_scores[word] = word_tfidf[word] * math.sqrt(chi_squares[word])
    
    # 計算每個詞彙與股價變動的相關性
    print("計算詞彙與股價變動的相關性...")
    
    # 創建詞彙-文檔矩陣
    word_list = list(word_scores.keys())
    word_dict = {word: idx for idx, word in enumerate(word_list)}
    
    print(f"構建詞彙-文檔矩陣，詞彙數: {len(word_list)}")
    # 計算每個詞彙在每篇文章中的出現次數
    word_doc_matrix = np.zeros((len(df), len(word_list)))
    
    for i, words in tqdm(enumerate(article_words_list), desc="構建詞彙-文檔矩陣", total=len(article_words_list)):
        for word in words:
            if word in word_dict:
                word_doc_matrix[i, word_dict[word]] += 1
    
    # 計算每個詞彙與不同時間點股價變動的相關性
    time_points = ['1d', '5d', '10d', '20d']
    correlations = {}
    
    for tp in time_points:
        print(f"計算與 {tp} 股價變動的相關性...")
        
        # 使用連續範圍的股價變動（如果有）
        if tp != '1d' and f'stock_trend_{tp}_continuous' in df.columns:
            price_changes = df[f'stock_trend_{tp}_continuous'].fillna(0).values
        else:
            # 否則使用原始股價變動
            price_changes = df[f'stock_trend_{tp}'].fillna(0).values
        
        # 計算每個詞彙與股價變動的相關性
        corr_values = []
        for j in tqdm(range(len(word_list)), desc=f"計算 {tp} 相關性"):
            word_presence = word_doc_matrix[:, j]
            
            # 只在詞彙出現的文章中計算相關性
            mask = word_presence > 0
            if np.sum(mask) > 5:  # 至少出現在 5 篇文章中
                # 檢查是否有標準差為零的情況
                if np.std(word_presence[mask]) > 0 and np.std(price_changes[mask]) > 0:
                    corr = np.corrcoef(word_presence[mask], price_changes[mask])[0, 1]
                    corr_values.append(corr if not np.isnan(corr) else 0)
                else:
                    corr_values.append(0)  # 如果標準差為零，相關係數設為零
            else:
                corr_values.append(0)
        
        correlations[tp] = corr_values
    
    # 創建結果 DataFrame
    print("創建結果 DataFrame...")
    result_df = pd.DataFrame({'Term': word_list})
    
    # 添加 TF-IDF 值
    result_df['TF_IDF'] = [word_tfidf[word] for word in word_list]
    
    # 添加情感類型 (根據與 1d 股價變動的相關性)
    sentiment_types = []
    for corr in correlations['1d']:
        if corr > 0.1:
            sentiment_types.append('Positive')
        elif corr < -0.1:
            sentiment_types.append('Negative')
        else:
            sentiment_types.append('Neutral')
    
    result_df['Type'] = sentiment_types
    
    # 添加詞彙長度
    result_df['N'] = [len(word) for word in word_list]
    
    # 添加相關性數據
    for tp in time_points:
        result_df[f'Corr_{tp}'] = correlations[tp]
    
    # 計算 TF 和 DF 在不同情感類型文章中的統計
    print("計算不同情感類型的統計數據...")
    
    # 根據 stock_trend_1d 將文章分為正面、中性和負面
    sentiment_labels = []
    for trend in df['stock_trend_1d'].fillna(0):
        if trend > 0.01:  # 上漲超過1%
            sentiment_labels.append('Positive')
        elif trend < -0.01:  # 下跌超過1%
            sentiment_labels.append('Negative')
        else:
            sentiment_labels.append('Neutral')
    
    # 統計不同情感類型中的 TF 和 DF
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        print(f"計算 {sentiment} 類型的統計...")
        # 計算 TF
        tf_values = []
        df_values = []
        
        for j, word in tqdm(enumerate(word_list), desc=f"計算 {sentiment} 統計", total=len(word_list)):
            tf = 0
            df_docs = set()
            
            for i, words in enumerate(article_words_list):
                if sentiment_labels[i] == sentiment:
                    word_count = words.count(word)
                    tf += word_count
                    if word_count > 0:
                        df_docs.add(i)
            
            tf_values.append(tf)
            df_values.append(len(df_docs))
        
        result_df[f'TF_{sentiment}'] = tf_values
        result_df[f'DF_{sentiment}'] = df_values
    
    # 計算卡方統計量 (TF)
    print("計算卡方統計量 (TF)...")
    total_pos_tf = result_df['TF_Positive'].sum()
    total_neg_tf = result_df['TF_Negative'].sum()
    total_tf = total_pos_tf + total_neg_tf
    
    # 計算每個詞的卡方值 (TF)
    result_df['Chi2_TF'] = 0
    for idx, row in tqdm(result_df.iterrows(), desc="計算卡方TF", total=len(result_df)):
        # 觀察到的頻率
        o11 = row['TF_Positive']  # 詞在正面文章中的頻率
        o12 = row['TF_Negative']  # 詞在負面文章中的頻率
        o_1 = o11 + o12  # 詞的總頻率
        
        # 期望頻率
        e11 = o_1 * (total_pos_tf / total_tf) if total_tf > 0 else 0
        e12 = o_1 * (total_neg_tf / total_tf) if total_tf > 0 else 0
        
        # 計算卡方值
        if e11 > 0 and e12 > 0:
            chi2 = ((o11 - e11) ** 2 / e11) + ((o12 - e12) ** 2 / e12)
            result_df.at[idx, 'Chi2_TF'] = chi2
    
    # 計算卡方統計量 (DF)
    print("計算卡方統計量 (DF)...")
    total_pos_df = result_df['DF_Positive'].max()  # 假設這是正面文章的總數
    total_neg_df = result_df['DF_Negative'].max()  # 假設這是負面文章的總數
    total_df = total_pos_df + total_neg_df
    
    # 計算每個詞的卡方值 (DF)
    result_df['Chi2_DF'] = 0
    for idx, row in tqdm(result_df.iterrows(), desc="計算卡方DF", total=len(result_df)):
        # 觀察到的文檔頻率
        o11 = row['DF_Positive']  # 詞出現在正面文章的數量
        o12 = row['DF_Negative']  # 詞出現在負面文章的數量
        o_1 = o11 + o12  # 詞出現的總文章數
        
        # 期望文檔頻率
        e11 = o_1 * (total_pos_df / total_df) if total_df > 0 else 0
        e12 = o_1 * (total_neg_df / total_df) if total_df > 0 else 0
        
        # 計算卡方值
        if e11 > 0 and e12 > 0:
            chi2 = ((o11 - e11) ** 2 / e11) + ((o12 - e12) ** 2 / e12)
            result_df.at[idx, 'Chi2_DF'] = chi2
    
    # 計算最終分數: TF-IDF * 卡方TF開根號 * 卡方DF開根號
    print("計算最終分數...")
    result_df['Score'] = result_df['TF_IDF'] * np.sqrt(result_df['Chi2_TF']) * np.sqrt(result_df['Chi2_DF'])
    
    # 按分數排序
    result_df = result_df.sort_values('Score', ascending=False)
    
    end_time = time.time()
    print(f"分析完成，總耗時: {(end_time - start_time) / 60:.2f} 分鐘")
    
    return result_df

def filter_terms_by_criteria(term_df, time_point, score_threshold=8, price_change_threshold=0.01):
    """根據指定條件篩選詞彙"""
    print(f"篩選 {time_point} 時間點的詞彙...")
    
    # 顯示分數的分布情況
    print(f"分數統計: min={term_df['Score'].min()}, max={term_df['Score'].max()}, mean={term_df['Score'].mean()}")
    
    # 根據分數分布情況，可能需要調整閾值
    adjusted_threshold = score_threshold
    if term_df['Score'].max() < score_threshold:
        # 如果最大分數小於閾值，調整閾值為最大分數的 80%
        adjusted_threshold = term_df['Score'].max() * 0.8
        print(f"調整分數閾值為 {adjusted_threshold}")
    
    # 顯示篩選前的總詞彙數
    print(f"篩選前總詞彙數: {len(term_df)}")
    
    # 根據分數篩選詞彙
    score_filtered = term_df[term_df['Score'] > adjusted_threshold]
    print(f"Score > {adjusted_threshold} 的詞彙數: {len(score_filtered)}")
    
    # 根據股價變動篩選詞彙
    corr_column = f'Corr_{time_point}'
    if corr_column in term_df.columns:
        # 篩選出股價變動超過閾值的詞彙
        price_filtered = term_df[abs(term_df[corr_column]) > price_change_threshold]
        print(f"股價變動 > {price_change_threshold*100}% 的詞彙數: {len(price_filtered)}")
        
        # 合併兩種篩選結果
        combined_filtered = pd.concat([score_filtered, price_filtered]).drop_duplicates('Term')
        print(f"合併後的詞彙數: {len(combined_filtered)}")
    else:
        print(f"警告: 找不到 {corr_column} 欄位，只使用分數篩選")
        combined_filtered = score_filtered
    
    return combined_filtered

def count_terms_in_document(document, terms):
    """計算文檔中每個詞出現的次數"""
    term_counts = {}
    
    if not isinstance(document, str):
        return term_counts
    
    # 對於每個詞，計算在文檔中出現的次數
    for term in terms:
        count = document.count(term)
        if count > 0:
            term_counts[term] = count
    
    return term_counts


def build_vector_space(articles_df, terms_df, time_point):
    """為每篇文章建立向量空間"""
    start_time = time.time()
    print(f"為 {time_point} 時間點建立向量空間...")
    
    # 設定 jieba 分詞
    stopwords = setup_jieba()
    
    # 從詞彙 DataFrame 中提取詞彙列表
    terms = terms_df['Term'].tolist()
    
    # 建立詞彙字典
    term_dict = {term: idx for idx, term in enumerate(terms)}
    
    # 保存詞彙字典
    dict_filename = f'jieba_{time_point}_trend_dictionary.json'
    with open(dict_filename, 'w', encoding='utf-8') as f:
        json.dump(term_dict, f, ensure_ascii=False)
    print(f"詞彙字典已保存到 {dict_filename}")
    
    # 為每篇文章建立向量
    rows = []
    cols = []
    data = []
    
    # 保存每篇文章的詞彙分數
    article_term_scores = []
    
    for doc_idx, content in enumerate(tqdm(articles_df['content'].fillna(''), desc="建立文章向量")):
        # 預處理文本
        preprocessed_text = preprocess_text(content)
        
        # 計算每個詞在文章中出現的次數
        term_counts = count_terms_in_document(preprocessed_text, terms)
        
        # 記錄該文章的詞彙分數
        article_score = {}
        
        for term, count in term_counts.items():
            if term in term_dict:
                term_idx = term_dict[term]
                rows.append(doc_idx)
                cols.append(term_idx)
                data.append(count)
                
                # 詞彙分數 = 該詞在文章中出現的次數
                article_score[term] = count
        
        article_term_scores.append(article_score)
    
    # 創建稀疏矩陣
    print("創建稀疏矩陣...")
    vector_matrix = csr_matrix((data, (rows, cols)), 
                              shape=(len(articles_df), len(terms)))
    
    # 保存向量模型
    vector_filename = f'jieba_{time_point}_trend_vector_model.npz'
    save_npz(vector_filename, vector_matrix)
    print(f"向量模型已保存到 {vector_filename}")
    
    # 保存每篇文章的詞彙分數
    scores_filename = f'jieba_{time_point}_article_term_scores.json'
    with open(scores_filename, 'w', encoding='utf-8') as f:
        json.dump(article_term_scores, f, ensure_ascii=False)
    print(f"文章詞彙分數已保存到 {scores_filename}")
    
    # 保存發文時間（只保留日期部分）
    if 'post_time' in articles_df.columns:
        # 將時間字串轉換為日期
        post_dates = []
        for post_time in articles_df['post_time'].fillna(''):
            try:
                # 假設時間格式為 'YYYY-MM-DD HH:MM:SS'
                date_str = post_time.split(' ')[0] if isinstance(post_time, str) else ''
                post_dates.append(date_str)
            except:
                post_dates.append('')
        
        # 保存日期到檔案
        post_time_filename = f'jieba_{time_point}_post_time.csv'
        np.savetxt(post_time_filename, post_dates, delimiter=',', fmt='%s')
        print(f"發文日期已保存到 {post_time_filename}")
    else:
        print("警告: 找不到 post_time 欄位")
        post_time_filename = None
    
    # 準備標籤資料
    trend_column = f'stock_trend_{time_point}'
    if time_point != '1d' and f'stock_trend_{time_point}_continuous' in articles_df.columns:
        # 使用連續範圍計算的股價變動
        trend_column = f'stock_trend_{time_point}_continuous'
    
    if trend_column in articles_df.columns:
        y_data = articles_df[trend_column].fillna(0).values
        
        # 保存標籤資料，使用浮點數格式以保留連續值
        y_filename = f'jieba_{time_point}_trend_y.csv'
        np.savetxt(y_filename, y_data, delimiter=',', fmt='%.6f')
        print(f"標籤資料已保存到 {y_filename}")
    else:
        print(f"警告: 找不到 {trend_column} 欄位，無法產生標籤資料")
        y_filename = None
    
    end_time = time.time()
    print(f"向量空間建立完成，耗時: {(end_time - start_time) / 60:.2f} 分鐘")
    
    return {
        'dictionary': dict_filename,
        'vector_model': vector_filename,
        'term_scores': scores_filename,
        'post_time': post_time_filename,
        'labels': y_filename,
        'term_count': len(terms)
    }


def process_time_point(term_df, articles_df, time_point, score_threshold=8, price_change_threshold=0.01):
    """處理特定時間點的資料"""
    start_time = time.time()
    print(f"\n===== 處理 {time_point} 時間點的資料 =====")
    
    # 篩選詞彙
    selected_terms_df = filter_terms_by_criteria(
        term_df, 
        time_point, 
        score_threshold, 
        price_change_threshold
    )
    
    # 按詞彙長度分組
    print("按詞彙長度分組...")
    for n in range(2, 7):
        n_terms = selected_terms_df[selected_terms_df['N'] == n]
        print(f"{n} 字詞彙數: {len(n_terms)}")
    
    # 建立向量空間
    result = build_vector_space(articles_df, selected_terms_df, time_point)
    
    # 保存篩選後的詞彙
    terms_filename = f'jieba_{time_point}_selected_terms.csv'
    selected_terms_df.to_csv(terms_filename, index=False)
    print(f"篩選後的詞彙已保存到 {terms_filename}")
    
    end_time = time.time()
    print(f"{time_point} 時間點處理完成，總耗時: {(end_time - start_time) / 60:.2f} 分鐘")
    
    return result

def analyze_ngram_files(ngram_files, min_df=5):
    """從 N-gram 檔案中分析詞彙特徵"""
    start_time = time.time()
    print("從 N-gram 檔案中分析詞彙特徵...")
    
    # 讀取所有 N-gram 檔案
    all_terms_df = []
    for file in ngram_files:
        print(f"讀取 {file}...")
        try:
            df = pd.read_csv(file)
            all_terms_df.append(df)
        except Exception as e:
            print(f"讀取 {file} 時出錯: {e}")
    
    print(f"讀取了 {len(all_terms_df)} 個檔案，共 {sum(len(df) for df in all_terms_df)} 個詞彙")
    
    # 合併所有 N-gram 的資料
    if not all_terms_df:
        print("錯誤：沒有成功讀取任何 N-gram 檔案")
        return None
    
    term_df = pd.concat(all_terms_df)
    
    # 檢查必要的欄位是否存在
    required_columns = ['Term', 'TF_IDF', 'Chi2_TF', 'Chi2_DF']
    missing_columns = [col for col in required_columns if col not in term_df.columns]
    
    if missing_columns:
        print(f"錯誤：缺少必要的欄位 {missing_columns}")
        
        # 如果缺少 TF_IDF，嘗試從 TF 和 DF 計算
        if 'TF_IDF' in missing_columns and 'TF' in term_df.columns and 'DF' in term_df.columns:
            print("嘗試從 TF 和 DF 計算 TF-IDF...")
            n_docs = term_df['DF'].max()  # 假設這是文章總數
            term_df['TF_IDF'] = term_df['TF'] * np.log(n_docs / (term_df['DF'] + 1))
            missing_columns.remove('TF_IDF')
        
        # 如果缺少 Chi2_TF，嘗試從 TF_Positive 和 TF_Negative 計算
        if 'Chi2_TF' in missing_columns and 'TF_Positive' in term_df.columns and 'TF_Negative' in term_df.columns:
            print("嘗試計算 Chi2_TF...")
            total_pos_tf = term_df['TF_Positive'].sum()
            total_neg_tf = term_df['TF_Negative'].sum()
            total_tf = total_pos_tf + total_neg_tf
            
            term_df['Chi2_TF'] = 0
            for idx, row in term_df.iterrows():
                o11 = row['TF_Positive']
                o12 = row['TF_Negative']
                o_1 = o11 + o12
                
                e11 = o_1 * (total_pos_tf / total_tf) if total_tf > 0 else 0
                e12 = o_1 * (total_neg_tf / total_tf) if total_tf > 0 else 0
                
                if e11 > 0 and e12 > 0:
                    chi2 = ((o11 - e11) ** 2 / e11) + ((o12 - e12) ** 2 / e12)
                    term_df.at[idx, 'Chi2_TF'] = chi2
            
            missing_columns.remove('Chi2_TF')
        
        # 如果缺少 Chi2_DF，嘗試從 DF_Positive 和 DF_Negative 計算
        if 'Chi2_DF' in missing_columns and 'DF_Positive' in term_df.columns and 'DF_Negative' in term_df.columns:
            print("嘗試計算 Chi2_DF...")
            total_pos_df = term_df['DF_Positive'].max()
            total_neg_df = term_df['DF_Negative'].max()
            total_df = total_pos_df + total_neg_df
            
            term_df['Chi2_DF'] = 0
            for idx, row in term_df.iterrows():
                o11 = row['DF_Positive']
                o12 = row['DF_Negative']
                o_1 = o11 + o12
                
                e11 = o_1 * (total_pos_df / total_df) if total_df > 0 else 0
                e12 = o_1 * (total_neg_df / total_df) if total_df > 0 else 0
                
                if e11 > 0 and e12 > 0:
                    chi2 = ((o11 - e11) ** 2 / e11) + ((o12 - e12) ** 2 / e12)
                    term_df.at[idx, 'Chi2_DF'] = chi2
            
            missing_columns.remove('Chi2_DF')
    
    # 檢查是否還有缺少的欄位
    if missing_columns:
        print(f"警告：仍然缺少欄位 {missing_columns}，將使用替代計算方法")
        
        # 如果仍然缺少必要欄位，使用替代方法計算分數
        if 'Score' not in term_df.columns:
            print("計算替代分數...")
            
            # 檢查哪些欄位可用
            if 'TF_IDF' in term_df.columns and 'Chi2_TF' in term_df.columns and 'Chi2_DF' in term_df.columns:
                # 使用完整公式
                term_df['Score'] = term_df['TF_IDF'] * np.sqrt(term_df['Chi2_TF']) * np.sqrt(term_df['Chi2_DF'])
                print("使用完整公式計算分數")
            elif 'TF_IDF' in term_df.columns:
                # 只使用 TF-IDF
                term_df['Score'] = term_df['TF_IDF']
                print("只使用 TF-IDF 計算分數")
            elif 'Corr_1d' in term_df.columns:
                # 使用相關係數
                term_df['Score'] = abs(term_df['Corr_1d']) * 10
                print("使用相關係數計算分數")
            else:
                # 設置預設分數
                term_df['Score'] = 1
                print("使用預設分數")
    
    end_time = time.time()
    print(f"N-gram 分析完成，總耗時: {(end_time - start_time) / 60:.2f} 分鐘")
    
    return term_df

def main():
    total_start_time = time.time()
    print("開始使用 N-gram 建立向量空間...")
    
    # 設定文件路徑
    ngram_files = [
        'character_2gram_analysis_100_result.csv',
        'character_3gram_analysis_100_result.csv',
        'character_4gram_analysis_100_result.csv',
        'character_5gram_analysis_100_result.csv',
        'character_6gram_analysis_100_result.csv'
    ]
    articles_file = 'marked_articles_innolux.csv'  # 您的文章數據文件
    
    # 檢查文件是否存在
    for file in ngram_files:
        if not os.path.exists(file):
            print(f"警告: 找不到文件 {file}")
    
    if not os.path.exists(articles_file):
        print(f"錯誤: 找不到文章文件 {articles_file}")
        return
    
    # 讀取文章資料
    print("讀取文章資料...")
    articles_df = pd.read_csv(articles_file)
    print(f"共讀取 {len(articles_df)} 篇文章")
    
    # 選擇分析方法
    if ngram_files:
        # 從 N-gram 檔案中分析詞彙特徵
        term_df = analyze_ngram_files(ngram_files)
        if term_df is None:
            print("錯誤：無法從 N-gram 檔案中分析詞彙特徵，改用直接分析文章")
            term_df = analyze_articles(articles_file, min_df=5)
    else:
        # 直接分析文章，提取詞彙特徵
        term_df = analyze_articles(articles_file, min_df=5)
    
    # 保存詞彙分析結果
    term_analysis_file = 'term_analysis_result.csv'
    term_df.to_csv(term_analysis_file, index=False)
    print(f"詞彙分析結果已保存到 {term_analysis_file}")
    
    # 處理各個時間點
    time_points = ['1d', '5d', '10d', '20d']
    results = {}
    
    # 設定閾值
    score_threshold = 8  # 分數閾值
    price_change_threshold = 0.01  # 股價變動閾值 (1%)
    
    for time_point in time_points:
        result = process_time_point(
            term_df, 
            articles_df, 
            time_point, 
            score_threshold,
            price_change_threshold
        )
        results[time_point] = result
    
    # 輸出處理結果摘要
    print("\n===== 處理結果摘要 =====")
    for time_point, result in results.items():
        print(f"{time_point} 時間點:")
        print(f"  - 詞彙字典: {result['dictionary']} (包含 {result['term_count']} 個詞)")
        print(f"  - 向量模型: {result['vector_model']}")
        print(f"  - 詞彙分數: {result['term_scores']}")
        if result['labels']:
            print(f"  - 標籤資料: {result['labels']}")
        else:
            print("  - 標籤資料: 無")
    
    total_end_time = time.time()
    total_minutes = (total_end_time - total_start_time) / 60
    print(f"\n全部處理完成，總耗時: {total_minutes:.2f} 分鐘 ({total_minutes/60:.2f} 小時)")

if __name__ == "__main__":
    main()
