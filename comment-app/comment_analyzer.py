import google.generativeai as genai
import pandas as pd
import json
import os
from dotenv import load_dotenv
from typing import Dict, List, Any
import time

class CommentAnalyzer:
    def __init__(self):
        """コメント分析器の初期化"""
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルに設定してください。")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def analyze_comment(self, comment: str) -> Dict[str, Any]:
        """
        単一のコメントを分析
        
        Args:
            comment (str): 分析対象のコメント
            
        Returns:
            Dict[str, Any]: 分析結果
        """
        if not comment or pd.isna(comment) or comment.strip() == "":
            return {
                "sentiment": "neutral",
                "category": "その他",
                "importance_score": 0,
                "risk_level": "low",
                "summary": "",
                "keywords": []
            }
        
        prompt = f"""
以下の講義アンケートのコメントを分析してください。
JSON形式で回答してください。

コメント: "{comment}"

以下の項目について分析してください：

1. sentiment: ポジティブ（positive）、ネガティブ（negative）、中立（neutral）のいずれか
2. category: 講義内容（content）、講義資料（materials）、運営（management）、その他（others）のいずれか
3. importance_score: 1-10の重要度スコア（具体性・緊急性・共通性を考慮）
4. risk_level: high（重要・緊急）、medium（やや重要）、low（通常）のいずれか
5. summary: コメントの要約（20文字以内）
6. keywords: 重要なキーワード（最大5個の配列）

回答例：
{{
    "sentiment": "negative",
    "category": "content",
    "importance_score": 8,
    "risk_level": "high",
    "summary": "講義内容が難しすぎる",
    "keywords": ["難しい", "理解困難", "講義内容"]
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # JSONの抽出（```json```で囲まれている場合の処理）
            if "```json" in result_text:
                start = result_text.find("```json") + 7
                end = result_text.find("```", start)
                result_text = result_text[start:end].strip()
            elif "```" in result_text:
                start = result_text.find("```") + 3
                end = result_text.find("```", start)
                result_text = result_text[start:end].strip()
            
            result = json.loads(result_text)
            
            # 結果が辞書型であることを確認
            if not isinstance(result, dict):
                print(f"警告: JSONパース結果が辞書型ではありません: {type(result)}")
                return {
                    "sentiment": "neutral",
                    "category": "others",
                    "importance_score": 1,
                    "risk_level": "low",
                    "summary": "分析エラー",
                    "keywords": []
                }
            
            # 必須フィールドの確認と修正
            if 'keywords' in result and not isinstance(result['keywords'], list):
                result['keywords'] = []
            
            return result
            
        except Exception as e:
            print(f"コメント分析エラー: {e}")
            print(f"レスポンステキスト: {result_text if 'result_text' in locals() else 'N/A'}")
            return {
                "sentiment": "neutral",
                "category": "others",
                "importance_score": 1,
                "risk_level": "low",
                "summary": "分析エラー",
                "keywords": []
            }
    
    def analyze_comments_batch(self, comments: List[str], delay: float = 0.5) -> List[Dict[str, Any]]:
        """
        複数のコメントを一括分析
        
        Args:
            comments (List[str]): 分析対象のコメントリスト
            delay (float): API呼び出し間の遅延（秒）
            
        Returns:
            List[Dict[str, Any]]: 分析結果のリスト
        """
        results = []
        total = len(comments)
        start_time = time.time()
        
        for i, comment in enumerate(comments):
            if i > 0:
                time.sleep(delay)  # API レート制限対策
                
            result = self.analyze_comment(comment)
            result['original_comment'] = comment
            result['index'] = i
            results.append(result)
            
            # 進捗表示
            progress = (i + 1) / total
            if (i + 1) % 5 == 0 or i + 1 == total:
                # プログレスバーの作成
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                # 推定残り時間の計算
                elapsed_time = time.time() - start_time
                if i > 0 and elapsed_time > 0:
                    avg_time_per_comment = elapsed_time / (i + 1)
                    remaining_time = avg_time_per_comment * (total - i - 1)
                    eta_str = f" | 残り時間: {int(remaining_time//60)}分{int(remaining_time%60)}秒"
                else:
                    eta_str = ""
                
                # センチメント統計の計算
                sentiments = [r['sentiment'] for r in results[:i+1]]
                pos_count = sentiments.count('positive')
                neg_count = sentiments.count('negative')
                
                print(f"\r[{bar}] {i + 1}/{total} ({progress*100:.1f}%) | ポジティブ: {pos_count} | ネガティブ: {neg_count}{eta_str}", end="", flush=True)
                
                if i + 1 == total:
                    print()  # 最後に改行
        
        return results
    
    def generate_summary_report(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析結果のサマリーレポートを生成
        
        Args:
            analysis_results (List[Dict[str, Any]]): 分析結果リスト
            
        Returns:
            Dict[str, Any]: サマリーレポート
        """
        if not analysis_results:
            return {}
        
        total_comments = len(analysis_results)
        
        # センチメント集計
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for result in analysis_results:
            sentiment_counts[result.get("sentiment", "neutral")] += 1
        
        # カテゴリ集計
        category_counts = {"content": 0, "materials": 0, "management": 0, "others": 0}
        
        for result in analysis_results:
            category = result.get("category", "others")
            # カテゴリがすでに英語で返されているので、そのまま使用
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts["others"] += 1
        
        # 重要度の高いコメント（スコア7以上）
        high_importance = [r for r in analysis_results if r.get("importance_score", 0) >= 7]
        
        # 危険度の高いコメント
        high_risk = [r for r in analysis_results if r.get("risk_level") == "high"]
        
        return {
            "total_comments": total_comments,
            "sentiment_distribution": {
                "positive": {"count": sentiment_counts["positive"], "percentage": sentiment_counts["positive"]/total_comments*100},
                "negative": {"count": sentiment_counts["negative"], "percentage": sentiment_counts["negative"]/total_comments*100},
                "neutral": {"count": sentiment_counts["neutral"], "percentage": sentiment_counts["neutral"]/total_comments*100}
            },
            "category_distribution": {
                "content": {"count": category_counts["content"], "percentage": category_counts["content"]/total_comments*100},
                "materials": {"count": category_counts["materials"], "percentage": category_counts["materials"]/total_comments*100},
                "management": {"count": category_counts["management"], "percentage": category_counts["management"]/total_comments*100},
                "others": {"count": category_counts["others"], "percentage": category_counts["others"]/total_comments*100}
            },
            "high_importance_comments": len(high_importance),
            "high_risk_comments": len(high_risk),
            "top_high_risk_comments": sorted(high_risk, key=lambda x: x.get("importance_score", 0), reverse=True)[:10]
        }

def process_excel_file(file_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Excelファイルを処理してコメント分析を実行
    
    Args:
        file_path (str): 入力Excelファイルパス
        output_path (str): 出力CSVファイルパス（省略可）
        
    Returns:
        Dict[str, Any]: 処理結果
    """
    # Excelファイル読み込み
    df = pd.read_excel(file_path)
    
    # コメント列を特定（自由記述項目）
    comment_columns = [
        '【必須】本日の講義で学んだことを50文字以上で入力してください。',
        '（任意）本日の講義で特によかった部分について、具体的にお教えください。',
        '（任意）分かりにくかった部分や改善点などがあれば、具体的にお教えください。',
        '（任意）講師について、よかった点や不満があった点などについて、具体的にお教えください。',
        '（任意）今後開講してほしい講義・分野などがあればお書きください。',
        '（任意）ご自由にご意見をお書きください。'
    ]
    
    analyzer = CommentAnalyzer()
    all_results = []
    
    for col in comment_columns:
        if col in df.columns:
            print(f"\n{col} の分析を開始...")
            comments = df[col].dropna().tolist()
            
            if comments:
                results = analyzer.analyze_comments_batch(comments)
                for result in results:
                    result['column_name'] = col
                all_results.extend(results)
    
    # サマリーレポート生成
    summary = analyzer.generate_summary_report(all_results)
    
    # CSV出力
    if output_path:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n結果を {output_path} に保存しました。")
    
    return {
        "analysis_results": all_results,
        "summary_report": summary,
        "original_data_shape": df.shape
    }

if __name__ == "__main__":
    # テスト実行
    file_path = "data/Day1_アンケート_.xlsx"
    output_path = "analysis_results.csv"
    
    try:
        results = process_excel_file(file_path, output_path)
        print("\n=== 分析完了 ===")
        print(f"総コメント数: {results['summary_report']['total_comments']}")
        print(f"高重要度コメント: {results['summary_report']['high_importance_comments']}")
        print(f"高危険度コメント: {results['summary_report']['high_risk_comments']}")
        
    except Exception as e:
        print(f"エラー: {e}")
        print("APIキーが設定されているか確認してください。")