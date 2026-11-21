
import os
import pandas as pd
import glob
import re
from tqdm import tqdm
import unicodedata
import emoji

def clean_text(text):
    """
    清理文本：去除標點符號、英文、數字、表情符號等
    
    參數:
    text: 要清理的文本
    
    返回:
    str: 清理後的文本
    """
    if not isinstance(text, str):
        return ""
    
    # 去除表情符號
    text = emoji.replace_emoji(text, replace='')
    
    # 去除英文和數字
    text = re.sub(r'[a-zA-Z0-9]+', '', text)
    
    # 去除標點符號（包括中文標點）
    punctuation = r'[^\w\s]'
    text = re.sub(punctuation, '', text)
    
    # 去除額外的空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def search_and_process_keywords(directory_path="data", keywords=None, columns_to_search=None):
    """
    搜尋指定目錄下所有 CSV 檔案中包含關鍵字的文章，並進行文本處理
    
    參數:
    directory_path: CSV 檔案所在的目錄路徑
    keywords: 要搜尋的關鍵字列表
    columns_to_search: 要搜尋的欄位名稱列表，預設為 None (搜尋 title 和 content)
    
    返回:
    dict: 包含搜尋結果的字典
    """
    if keywords is None:
        keywords = ["群創", "洪進揚", "楊柱祥", "友達光電", "AUO", "華映", "CPT", "INNOLUX", "面板雙虎", "3481"]
    
    if columns_to_search is None:
        columns_to_search = ['title', 'content']
    
    # 取得目錄下所有 CSV 檔案
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        return {"error": f"在 {directory_path} 中找不到 CSV 檔案"}
    
    results = {
        "total_articles": 0,
        "matching_articles": 0,
        "files_processed": 0,
        "files_with_matches": 0,
        "file_details": [],
        "keyword_counts": {keyword: 0 for keyword in keywords},
        "processed_data": []  # 將處理後的資料存在這裡
    }
    
    # 為每個關鍵字編譯正則表達式模式，不區分大小寫
    patterns = {keyword: re.compile(re.escape(keyword), re.IGNORECASE) for keyword in keywords}
    
    # 處理每個 CSV 檔案
    for csv_file in tqdm(csv_files, desc="處理檔案"):
        try:
            print(f"正在處理檔案: {csv_file}")
            
            # 讀取 CSV 檔案
            df = pd.read_csv(csv_file)
            file_articles = len(df)
            results["total_articles"] += file_articles
            
            # 檢查是否為 PTT 檔案（檢查是否缺少 content_type 欄位）
            is_ptt = 'content_type' not in df.columns
            
            # 檢查欄位是否存在
            valid_columns = [col for col in columns_to_search if col in df.columns]
            if not valid_columns:
                results["file_details"].append({
                    "file": os.path.basename(csv_file),
                    "error": f"找不到指定的欄位 {columns_to_search}",
                    "total_articles": file_articles,
                    "matching_articles": 0
                })
                continue
            
            # 創建一個新的 DataFrame 來存儲結果
            matching_rows = []
            file_matches = 0
            
            # 逐行處理每篇文章
            for idx, row in df.iterrows():
                article_keywords = []
                
                # 檢查每個有效欄位
                for col in valid_columns:
                    # 確保欄位值是字串
                    if pd.notna(row.get(col)) and isinstance(row.get(col), str):
                        # 檢查每個關鍵字
                        for keyword, pattern in patterns.items():
                            if pattern.search(row[col]) and keyword not in article_keywords:
                                article_keywords.append(keyword)
                
                # 如果文章包含任何關鍵字
                if article_keywords:
                    file_matches += 1
                    
                    # 更新關鍵字計數
                    for keyword in article_keywords:
                        results["keyword_counts"][keyword] += 1
                    
                    # 處理文本內容
                    processed_row = row.copy()
                    
                    # 處理 title 和 content
                    if 'title' in processed_row and isinstance(processed_row['title'], str):
                        processed_row['title'] = clean_text(processed_row['title'])
                    
                    if 'content' in processed_row and isinstance(processed_row['content'], str):
                        processed_row['content'] = clean_text(processed_row['content'])
                    
                    # 添加來源檔案名稱和關鍵字資訊
                    processed_row['source_file'] = os.path.basename(csv_file)
                    processed_row['keywords'] = ', '.join(article_keywords)
                    
                    # 將處理後的行添加到結果中
                    matching_rows.append(processed_row)
            
            # 更新結果
            results["matching_articles"] += file_matches
            results["files_processed"] += 1
            
            if file_matches > 0:
                results["files_with_matches"] += 1
                
            results["file_details"].append({
                "file": os.path.basename(csv_file),
                "total_articles": file_articles,
                "matching_articles": file_matches
            })
            
            # 如果有匹配的行，創建 DataFrame 並添加到結果中
            if matching_rows:
                matching_df = pd.DataFrame(matching_rows)
                
                # 確保只保留需要的欄位
                required_fields = ["id", "p_type", "s_name", "s_area_name", "post_time", 
                                  "title", "author", "content", "page_url"]
                
                # 對於非 PTT 檔案，添加 content_type 欄位
                if not is_ptt:
                    required_fields.append("content_type")
                
                # 添加關鍵字和來源檔案欄位
                required_fields.extend(["keywords", "source_file"])
                
                # 只保留存在的欄位
                existing_fields = [field for field in required_fields if field in matching_df.columns]
                matching_df = matching_df[existing_fields]
                
                # 添加到結果中
                results["processed_data"].append(matching_df)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"處理檔案 {csv_file} 時發生錯誤: {str(e)}")
            print(error_details)
            results["file_details"].append({
                "file": os.path.basename(csv_file),
                "error": f"{str(e)}",
                "total_articles": file_articles if 'file_articles' in locals() else 0,
                "matching_articles": 0
            })
    
    # 計算匹配率
    if results["total_articles"] > 0:
        results["match_percentage"] = (results["matching_articles"] / results["total_articles"]) * 100
    else:
        results["match_percentage"] = 0
    
    return results

def display_results(results):
    """
    顯示搜尋結果
    
    參數:
    results: 搜尋結果字典
    """
    if "error" in results:
        print(f"錯誤: {results['error']}")
        return
    
    print("\n===== 搜尋結果摘要 =====")
    print(f"處理的檔案數: {results['files_processed']}")
    print(f"包含匹配的檔案數: {results['files_with_matches']}")
    print(f"總文章數: {results['total_articles']:,}")
    print(f"匹配的文章數: {results['matching_articles']:,}")
    print(f"匹配率: {results['match_percentage']:.2f}%")
    
    print("\n===== 關鍵字出現次數 =====")
    # 依出現次數排序關鍵字
    sorted_keywords = sorted(results["keyword_counts"].items(), key=lambda x: x[1], reverse=True)
    for keyword, count in sorted_keywords:
        print(f"'{keyword}': {count:,} 篇文章")
    
    print("\n===== 各檔案詳細資訊 =====")
    # 依匹配文章數排序
    sorted_details = sorted(results["file_details"], 
                           key=lambda x: x.get("matching_articles", 0) if "matching_articles" in x else -1,
                           reverse=True)
    
    for detail in sorted_details:
        if "error" in detail:
            print(f"檔案: {detail['file']} - 錯誤: {detail['error']}")
        else:
            match_rate = (detail["matching_articles"] / detail["total_articles"]) * 100 if detail["total_articles"] > 0 else 0
            print(f"檔案: {detail['file']} - 匹配文章: {detail['matching_articles']:,}/{detail['total_articles']:,} ({match_rate:.2f}%)")

def main():
    """主程式"""
    print("===== CSV 檔案關鍵字搜尋與文本處理工具 =====")
    print("預設搜尋目錄: data")
    print("預設搜尋欄位: title, content")
    print("預設關鍵字: 群創, 洪進揚, 楊柱祥, 友達光電, AUO, 華映, CPT, INNOLUX, 面板雙虎, 3481")
    
    # 確認目錄存在
    if not os.path.isdir("data"):
        print(f"錯誤: 目錄 'data' 不存在")
        return
    
    # 執行搜尋和處理
    print("\n正在搜尋關鍵字並處理文本...")
    results = search_and_process_keywords()
    
    # 顯示結果
    display_results(results)
    
    # 儲存處理後的資料
    if 'processed_data' in results and results['processed_data']:
        try:
            # 合併所有處理後的 DataFrame
            all_processed_data = pd.concat(results['processed_data'], ignore_index=True)
            
            # 儲存為 CSV 檔案
            output_file = "processed_articles.csv"
            all_processed_data.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n處理後的資料已儲存至 {output_file}")
            print(f"共 {len(all_processed_data):,} 篇文章")
            
            # 顯示預覽
            print("\n資料預覽:")
            print(all_processed_data.head(3).to_string())
            
        except Exception as e:
            print(f"儲存處理後的資料時發生錯誤: {e}")
    else:
        print("\n沒有找到匹配的文章，無法儲存處理後的資料")

if __name__ == "__main__":
    main()
