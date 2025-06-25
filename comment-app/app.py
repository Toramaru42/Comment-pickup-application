import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from comment_analyzer import CommentAnalyzer, process_excel_file
import os
import json
from datetime import datetime

# ページ設定
st.set_page_config(
    page_title="講義アンケート コメントピックアップアプリ",
    page_icon="📊",
    layout="wide"
)

# セッション状態の初期化
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'summary_report' not in st.session_state:
    st.session_state.summary_report = None

def main():
    st.title("📊 講義アンケート コメントピックアップアプリ")
    st.markdown("---")
    
    # サイドバー
    st.sidebar.title("設定")
    
    # APIキー設定
    api_key = st.sidebar.text_input(
        "Google Gemini APIキー",
        type="password",
        help="Google AI StudioからAPIキーを取得してください"
    )
    
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key
    
    # ファイルアップロード
    st.sidebar.markdown("### ファイルアップロード")
    uploaded_file = st.sidebar.file_uploader(
        "アンケートExcelファイル",
        type=['xlsx', 'xls'],
        help="アンケートデータが含まれたExcelファイルをアップロードしてください"
    )
    
    # メインコンテンツ
    tab1, tab2, tab3, tab4 = st.tabs(["📤 データアップロード", "📈 分析結果", "🔍 詳細分析", "📊 統計情報"])
    
    with tab1:
        st.header("データアップロード・分析")
        
        if uploaded_file is not None:
            # ファイル情報表示
            st.success(f"ファイル '{uploaded_file.name}' がアップロードされました")
            
            # プレビュー表示
            if st.button("📋 データプレビュー"):
                try:
                    df = pd.read_excel(uploaded_file)
                    st.subheader("データプレビュー")
                    st.write(f"データ形状: {df.shape[0]}行 × {df.shape[1]}列")
                    st.dataframe(df.head())
                    
                    # コメント列の確認
                    comment_columns = [col for col in df.columns if any(keyword in col for keyword in ['コメント', '意見', '感想', '要望', '改善', 'よかった', 'わかりにくかった'])]
                    if comment_columns:
                        st.subheader("検出されたコメント列")
                        for col in comment_columns:
                            non_null_count = df[col].count()
                            st.write(f"• {col}: {non_null_count}件のコメント")
                    
                except Exception as e:
                    st.error(f"ファイル読み込みエラー: {e}")
            
            # 分析実行
            st.markdown("### 🤖 AI分析実行")
            
            if not api_key:
                st.warning("⚠️ Google Gemini APIキーを入力してください")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    max_comments = st.number_input(
                        "最大分析コメント数",
                        min_value=10,
                        max_value=1000,
                        value=20,
                        help="分析するコメント数の上限（APIコスト節約のため）"
                    )
                
                with col2:
                    delay_time = st.number_input(
                        "API呼び出し間隔（秒）",
                        min_value=0.1,
                        max_value=5.0,
                        value=0.50,
                        step=0.1,
                        help="APIレート制限対策"
                    )
                
                if st.button("🚀 分析開始", type="primary"):
                    try:
                        # 進捗バーとステータス
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("ファイルを読み込み中...")
                        df = pd.read_excel(uploaded_file)
                        progress_bar.progress(0.1)
                        
                        # 一時ファイル保存
                        temp_file = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        df.to_excel(temp_file, index=False)
                        
                        status_text.text("AI分析を実行中...")
                        progress_bar.progress(0.3)
                        
                        # 分析実行
                        analyzer = CommentAnalyzer()
                        comment_columns = [
                            '【必須】本日の講義で学んだことを50文字以上で入力してください。',
                            '（任意）本日の講義で特によかった部分について、具体的にお教えください。',
                            '（任意）分かりにくかった部分や改善点などがあれば、具体的にお教えください。',
                            '（任意）講師について、よかった点や不満があった点などについて、具体的にお教えください。',
                            '（任意）今後開講してほしい講義・分野などがあればお書きください。',
                            '（任意）ご自由にご意見をお書きください。'
                        ]
                        
                        all_results = []
                        total_columns = len([col for col in comment_columns if col in df.columns])
                        processed_columns = 0
                        
                        for col in comment_columns:
                            if col in df.columns:
                                status_text.text(f"分析中: {col}")
                                comments = df[col].dropna().head(max_comments).tolist()
                                
                                if comments:
                                    results = analyzer.analyze_comments_batch(comments, delay=delay_time)
                                    for result in results:
                                        result['column_name'] = col
                                    all_results.extend(results)
                                
                                processed_columns += 1
                                progress_bar.progress(min(0.3 + (processed_columns / total_columns) * 0.6, 1.0))
                        
                        status_text.text("サマリーレポートを生成中...")
                        progress_bar.progress(0.9)
                        
                        # サマリー生成
                        summary = analyzer.generate_summary_report(all_results)
                        
                        # セッション状態に保存
                        st.session_state.analysis_results = all_results
                        st.session_state.summary_report = summary
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ 分析完了!")
                        
                        # 一時ファイル削除
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        
                        st.success(f"🎉 分析が完了しました！総コメント数: {len(all_results)}")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"分析エラー: {e}")
                        if 'temp_file' in locals() and os.path.exists(temp_file):
                            os.remove(temp_file)
    
    with tab2:
        st.header("分析結果概要")
        
        if st.session_state.summary_report:
            summary = st.session_state.summary_report
            
            # KPI表示
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "総コメント数",
                    summary['total_comments']
                )
            
            with col2:
                st.metric(
                    "高重要度コメント",
                    summary['high_importance_comments'],
                    delta=f"{summary['high_importance_comments']/summary['total_comments']*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    "高危険度コメント",
                    summary['high_risk_comments'],
                    delta="要注意" if summary['high_risk_comments'] > 0 else "正常"
                )
            
            with col4:
                positive_rate = summary['sentiment_distribution']['positive']['percentage']
                st.metric(
                    "ポジティブ率",
                    f"{positive_rate:.1f}%"
                )
            
            # センチメント分析結果
            st.subheader("😊 センチメント分析")
            col1, col2 = st.columns(2)
            sentiment_data = summary.get('sentiment_distribution', {})

            # 日本語表示のためのマッピング
            SENTIMENT_JP_MAP = {
                'positive': 'ポジティブ',
                'negative': 'ネガティブ',
                'neutral': '中立'
            }
            # 存在するキーのみを対象にする
            sentiment_labels = [SENTIMENT_JP_MAP.get(key, key) for key in sentiment_data.keys()]
            sentiment_values = [data.get('count', 0) for data in sentiment_data.values()]
            

            with col1:
                # 円グラフ
                if sentiment_values:
                    fig_pie = px.pie(
                        values=sentiment_values,
                        names=sentiment_labels, # 日本語キーを直接使用
                        title="センチメント分布",
                        color_discrete_map={'ポジティブ': '#00CC96', 'ネガティブ': '#EF553B', '中立': '#AB63FA'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.write("カテゴリデータがありません。")
            
            with col2:
                # バーチャート
                if sentiment_data:
                    sentiment_list = []
                    for key, data in sentiment_data.items():
                        sentiment_list.append({
                            'センチメント': SENTIMENT_JP_MAP.get(key, key),
                            '件数': data.get('count', 0),
                            '割合(%)': data.get('percentage', 0.0)
                            })
                    sentiment_df = pd.DataFrame(sentiment_list)
                    st.dataframe(
                        sentiment_df,
                        use_container_width=True,
                        column_config={"割合(%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)}
                        )
                else:
                    st.write("センチメントデータがありません。")
            
            # カテゴリ分析結果
            st.subheader("📂 カテゴリ分析")
            category_data = summary.get('category_distribution', {})
            category_names = list(category_data.keys())
            category_values = [data.get('count', 0) for data in category_data.values()]

            col1, col2 = st.columns(2)
            
            with col1:
                # カテゴリ円グラフ
                if category_values:
                    fig_cat = px.pie(
                    values=category_values,
                    names=category_labels,
                    title="カテゴリ分布"
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    st.write("カテゴリデータがありません。")

            with col2:
                # カテゴリデータフレーム
                if category_data:
                    category_list = []
                    for category_name, data in category_data.items():
                        category_list.append({
                            'カテゴリ': category_name, # 日本語キーを直接使用
                            '件数': data.get('count', 0),
                            '割合(%)': data.get('percentage', 0.0)
                        })
                    category_df = pd.DataFrame(category_list)
                    st.dataframe(
                        category_df,
                        use_container_width=True,
                        column_config={"割合(%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)}
                        )
                else:
                    st.write("カテゴリデータがありません。")
    
    with tab3:
        st.header("詳細分析・高危険度コメント")
        
        if st.session_state.analysis_results and st.session_state.summary_report:
            # 高危険度コメント表示
            high_risk_comments = st.session_state.summary_report.get('top_high_risk_comments', [])
            
            if high_risk_comments:
                st.subheader("🚨 高危険度コメント（上位10件）")
                
                for i, comment in enumerate(high_risk_comments, 1):
                    with st.expander(f"#{i} {comment.get('summary', 'N/A')} (重要度: {comment.get('importance_score', 0)})"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write("**元のコメント:**")
                            st.write(comment.get('original_comment', 'N/A'))
                            
                            keywords = comment.get('keywords', [])
                            if keywords and isinstance(keywords, list):
                                st.write("**キーワード:**", ', '.join(keywords))
                        
                        with col2:
                            st.write(f"**センチメント:** {comment.get('sentiment', 'N/A')}")
                            st.write(f"**カテゴリ:** {comment.get('category', 'N/A')}")
                            st.write(f"**重要度:** {comment.get('importance_score', 0)}/10")
                            st.write(f"**危険度:** {comment.get('risk_level', 'N/A')}")
            
            # フィルタリング機能
            st.subheader("🔍 フィルタリング・検索")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_filter = st.selectbox(
                    "センチメント",
                    ["全て", "positive", "negative", "neutral"]
                )
            
            with col2:
                category_labels = {"全て": "全て", "講義内容": "content", "講義資料": "materials", "運営": "management", "その他": "others"}
                category_filter = st.selectbox(
                    "カテゴリ",
                    list(category_labels.keys())
                )
            
            with col3:
                min_importance = st.slider(
                    "最小重要度",
                    min_value=1,
                    max_value=10,
                    value=1
                )
            
            # フィルタ適用
            filtered_results = st.session_state.analysis_results.copy()
            
            if sentiment_filter != "全て":
                filtered_results = [r for r in filtered_results if r.get('sentiment') == sentiment_filter]
            
            if category_filter != "全て":
                category_value = category_labels[category_filter]
                filtered_results = [r for r in filtered_results if r.get('category') == category_value]
            
            filtered_results = [r for r in filtered_results if r.get('importance_score', 0) >= min_importance]
            
            st.write(f"フィルタ結果: {len(filtered_results)}件")
            
            # 結果表示
            if filtered_results:
                results_df = pd.DataFrame(filtered_results)
                st.dataframe(
                    results_df[['original_comment', 'sentiment', 'category', 'importance_score', 'summary', 'keywords']],
                    use_container_width=True
                )
        else:
            st.info("📤 まず分析を実行してください。")
    
    with tab4:
        st.header("統計情報・エクスポート")
        
        if st.session_state.analysis_results:
            # エクスポート機能
            st.subheader("📥 データエクスポート")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 CSV形式でダウンロード"):
                    results_df = pd.DataFrame(st.session_state.analysis_results)
                    csv = results_df.to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label="💾 CSVファイルをダウンロード",
                        data=csv,
                        file_name=f"comment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 JSONレポートをダウンロード"):
                    json_data = {
                        "analysis_results": st.session_state.analysis_results,
                        "summary_report": st.session_state.summary_report,
                        "generated_at": datetime.now().isoformat()
                    }
                    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="💾 JSONファイルをダウンロード",
                        data=json_str,
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            # 統計サマリー
            st.subheader("📈 詳細統計")
            
            if st.session_state.summary_report:
                summary = st.session_state.summary_report
                
                # 重要度分布
                importance_scores = [r.get('importance_score', 0) for r in st.session_state.analysis_results]
                fig_hist = px.histogram(
                    x=importance_scores,
                    title="重要度スコア分布",
                    labels={'x': '重要度スコア', 'y': '件数'},
                    nbins=10
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # 推奨アクション
                st.subheader("💡 推奨アクション")
                
                negative_rate = summary['sentiment_distribution']['negative']['percentage']
                high_risk_rate = summary['high_risk_comments'] / summary['total_comments'] * 100
                
                if negative_rate > 30:
                    st.warning(f"⚠️ ネガティブコメントが{negative_rate:.1f}%と高めです。改善項目の検討をお勧めします。")
                
                if high_risk_rate > 5:
                    st.error(f"🚨 高危険度コメントが{high_risk_rate:.1f}%あります。緊急対応が必要です。")
                
                if negative_rate < 20 and high_risk_rate < 3:
                    st.success("✅ 全体的に良好な評価です。現在の取り組みを継続してください。")
        
        else:
            st.info("📤 まず分析を実行してください。")

if __name__ == "__main__":
    main()