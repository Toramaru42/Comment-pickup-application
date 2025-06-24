import pandas as pd
import os

def analyze_excel_file(file_path):
    """Excelファイルを分析してデータ構造を確認"""
    if not os.path.exists(file_path):
        print(f"ファイルが見つかりません: {file_path}")
        return
    
    try:
        # Excelファイルの全シートを読み込み
        excel_file = pd.ExcelFile(file_path)
        print(f"ファイル名: {os.path.basename(file_path)}")
        print(f"シート数: {len(excel_file.sheet_names)}")
        print(f"シート名: {excel_file.sheet_names}")
        print("-" * 50)
        
        # 各シートの内容を確認
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"\nシート名: {sheet_name}")
            print(f"行数: {len(df)}")
            print(f"列数: {len(df.columns)}")
            print(f"列名: {list(df.columns)}")
            
            # 最初の数行を表示
            print("\n最初の3行:")
            print(df.head(3))
            
            # 各列のデータ型を確認
            print("\nデータ型:")
            print(df.dtypes)
            
            # 空の値の確認
            print("\n欠損値の数:")
            print(df.isnull().sum())
            
            print("-" * 50)
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    # data/フォルダ内のファイルを分析
    data_dir = "data"
    
    # Day1のファイルを分析
    file_path = os.path.join(data_dir, "Day1_アンケート_.xlsx")
    analyze_excel_file(file_path)