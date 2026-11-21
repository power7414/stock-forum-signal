
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import sys

# 顯示 Python 和套件版本，以便診斷
print(f"Python 版本: {sys.version}")
print(f"Pandas 版本: {pd.__version__}")

# 忽略警告
warnings.filterwarnings('ignore')

# 設定檔案路徑 - 請修改為您的實際路徑
file_path = 'bda2025_stock_202301-202503_上市.xlsx'

# 檢查檔案是否存在
if not os.path.exists(file_path):
    print(f"檔案 {file_path} 不存在，請確認檔案路徑")
    exit(1)

# 嘗試安裝或更新必要的套件
try:
    import pip
    print("嘗試更新必要的套件...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pandas", "openpyxl", "xlrd"])
    print("套件更新完成")
except Exception as e:
    print(f"更新套件時發生錯誤: {str(e)}")

# 讀取所有工作表
sheet_names = ['2023H1', '2023H2', '2024H1', '2024H2', '2025']
all_data = []

# 嘗試使用不同的方法讀取 Excel 檔案
try:
    # 方法 1: 使用 xlrd 引擎
    print("嘗試使用 xlrd 引擎讀取檔案...")
    excel_data = {}
    for sheet in sheet_names:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet, engine='xlrd')
            excel_data[sheet] = df
            print(f"成功使用 xlrd 讀取工作表 {sheet}")
        except Exception as e:
            print(f"使用 xlrd 讀取工作表 {sheet} 失敗: {str(e)}")
except Exception as e:
    print(f"使用 xlrd 引擎讀取失敗: {str(e)}")
    excel_data = {}

# 如果 xlrd 失敗，嘗試使用 openpyxl
if not excel_data:
    try:
        print("嘗試使用 openpyxl 引擎讀取檔案...")
        excel_data = {}
        for sheet in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl')
                excel_data[sheet] = df
                print(f"成功使用 openpyxl 讀取工作表 {sheet}")
            except Exception as e:
                print(f"使用 openpyxl 讀取工作表 {sheet} 失敗: {str(e)}")
    except Exception as e:
        print(f"使用 openpyxl 引擎讀取失敗: {str(e)}")

# 如果上述方法都失敗，嘗試最後一種方法
if not excel_data:
    try:
        print("嘗試直接讀取整個 Excel 檔案...")
        excel_data = pd.read_excel(file_path, sheet_name=None)
        print(f"成功讀取 Excel 檔案，包含工作表: {list(excel_data.keys())}")
    except Exception as e:
        print(f"讀取 Excel 檔案失敗: {str(e)}")
        print("無法讀取 Excel 檔案，請檢查檔案格式或嘗試轉換為 CSV 格式")
        exit(1)

# 處理每個工作表中的資料
for sheet, df in excel_data.items():
    try:
        print(f"\n處理工作表: {sheet}")
        print(f"工作表 {sheet} 的欄位: {list(df.columns)}")
        
        # 檢查是否有必要的欄位
        required_columns = ['代號', '名稱', '年月日', '開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)', '成交量(千股)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"工作表 {sheet} 缺少以下欄位: {', '.join(missing_columns)}")
            continue
        
        # 顯示工作表中的唯一代號，以便檢查
        unique_codes = df['代號'].unique()
        print(f"工作表 {sheet} 中的代號有: {unique_codes[:10]}{'...' if len(unique_codes) > 10 else ''}")
        
        # 篩選出群創的資料
        # 先將代號轉換為字串，以便比較
        df['代號'] = df['代號'].astype(str)
        innolux_data = df[df['代號'] == '3481']
        
        if innolux_data.empty:
            print(f"工作表 {sheet} 中沒有找到群創(3481)的資料")
            continue
            
        all_data.append(innolux_data)
        print(f"已成功讀取工作表 {sheet} 中的群創資料，共 {len(innolux_data)} 筆")
        
    except Exception as e:
        print(f"處理工作表 {sheet} 時發生錯誤: {str(e)}")

# 合併所有資料
if all_data:
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\n合併後共有 {len(combined_data)} 筆群創資料")
    
    # 確保日期欄位是日期格式
    combined_data['年月日'] = pd.to_datetime(combined_data['年月日'], errors='coerce')
    
    # 按日期排序
    combined_data = combined_data.sort_values('年月日')
    
    # 儲存為CSV檔案
    csv_path = 'data/innolux_stock_data.csv'
    combined_data.to_csv(csv_path, index=False)
    print(f"已將群創股價資料儲存至: {csv_path}")
    
    # 繪製歷史股價圖
    plt.figure(figsize=(12, 8))
    
    # 繪製收盤價
    plt.subplot(2, 1, 1)
    plt.plot(combined_data['年月日'], combined_data['收盤價(元)'], 'b-', label='收盤價')
    plt.title('群創(3481) 歷史收盤價', fontsize=14)
    plt.grid(True)
    plt.legend()
    
    # 繪製成交量
    plt.subplot(2, 1, 2)
    plt.bar(combined_data['年月日'], combined_data['成交量(千股)'], color='g', alpha=0.7, label='成交量(千股)')
    plt.title('群創(3481) 歷史成交量', fontsize=14)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # 儲存圖片
    plt.savefig('data/innolux_stock_chart.png')
    print("已儲存群創股價走勢圖至: data/innolux_stock_chart.png")
    
    # 顯示圖片
    plt.show()
    
else:
    print("\n沒有找到任何群創的股價資料")
    print("請檢查Excel檔案中是否包含群創(3481)的資料，或者代號格式是否正確")
    print("建議：手動打開Excel檔案，確認群創的代號格式和各工作表的內容")
