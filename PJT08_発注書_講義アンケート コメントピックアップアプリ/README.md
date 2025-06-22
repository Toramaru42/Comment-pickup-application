# 講義アンケート コメントピックアップアプリ

プロジェクトの詳細は、所定のドキュメントを確認すること。

## セットアップ・使用方法

### 1. 環境構築

```bash
# 仮想環境作成
uv venv

# 仮想環境有効化
source .venv/bin/activate

# 依存パッケージインストール
uv pip install pandas openpyxl google-generativeai streamlit python-dotenv plotly
```

### 2. APIキー設定

Google AI Studio（https://makersuite.google.com/app/apikey）からGemini APIキーを取得し、以下のいずれかの方法で設定：

**方法1: .envファイル作成**
```bash
cp .env.example .env
# .envファイルを編集してAPIキーを設定
GOOGLE_API_KEY=your_api_key_here
```

**方法2: Webアプリ上で直接入力**
- アプリ起動後、サイドバーでAPIキーを入力

### 3. アプリケーション起動

```bash
# 仮想環境が有効化されていることを確認
source .venv/bin/activate

# Streamlitアプリ起動
streamlit run app.py
```

### 4. 使用手順

1. **データアップロード**
   - Webブラウザでアプリにアクセス
   - サイドバーでAPIキーを入力
   - アンケートExcelファイル（.xlsx）をアップロード

2. **分析実行**
   - 分析するコメント数の上限を設定（デフォルト100件）
   - API呼び出し間隔を設定（デフォルト1秒）
   - 「分析開始」ボタンをクリック

3. **結果確認**
   - 分析結果タブで概要を確認
   - 詳細分析タブで高危険度コメントを確認
   - 統計情報タブでデータをエクスポート

### 5. ファイル構成

```
├── app.py                 # Streamlit Webアプリケーション
├── comment_analyzer.py    # コメント分析エンジン
├── analyze_data.py        # データ分析ユーティリティ
├── requirements.txt       # 依存パッケージリスト
├── .env.example          # 環境変数設定例
├── data/                 # サンプルアンケートデータ
│   ├── Day1_アンケート_.xlsx
│   ├── Day2_アンケート_.xlsx
│   └── ...
└── README.md             # このファイル
```

## 機能詳細

### コメント分析機能
- **センチメント分析**: ポジティブ/ネガティブ/中立を自動判定
- **カテゴリ分類**: 講義内容/講義資料/運営/その他に自動分類
- **重要度スコア**: 1-10段階で重要度を数値化
- **危険度判定**: 緊急対応が必要なコメントを特定

### 可視化機能
- センチメント分布の円グラフ・棒グラフ
- カテゴリ別統計情報
- 重要度スコア分布ヒストグラム
- 高危険度コメントの詳細表示

### エクスポート機能
- 分析結果のCSVダウンロード
- 詳細レポートのJSONダウンロード

## 注意事項

- Google Gemini APIの利用料金が発生します
- 大量のコメント分析時はAPI呼び出し回数に注意してください
- APIレート制限を避けるため、適切な間隔設定を推奨します

## 実装内容
- uv venvで仮想環境を構築し、必要なパッケージをインストール
- Google Gemini APIを使用したコメント分析システム
- StreamlitによるWebアプリケーション UI
- Excel形式のアンケートデータからコメントを自動分析・分類

### 使用方法

#### アプリ起動
source .venv/bin/activate
streamlit run app.py

アプリでは以下が可能です:
- Excelファイルのアップロードと分析
- ポジティブ/ネガティブ分類と重要度判定
- 高危険度コメントの特定
- 統計情報の可視化とCSV/JSONエクスポート