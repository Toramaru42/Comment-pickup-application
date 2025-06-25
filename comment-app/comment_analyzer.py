import google.generativeai as genai
import pandas as pd
import json
import os
from dotenv import load_dotenv
from typing import Dict, List, Any
import time
from collections import defaultdict

class CommentAnalyzer:
    def __init__(self):
        """コメント分析器の初期化"""
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルに設定してください。")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def _analyze_single_comment_phase1(self, comment: str) -> Dict[str, Any]:
        """
        【フェーズ1】単一のコメントを分析する（共通性は考慮しない）
        
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
あなたは大学の授業運営を支援する優秀なアシスタントです。
講義アンケートのコメントを以下のルールとコメント例を参考に分析してください。
指定されたJSON形式で出力してください。

# ルール
1. **Sentiment(感情分析)**:コメントが講座や講義に対して肯定的か、否定的か、中立的かを判断し、`positive`, `negative`, `neutral`のいずれかに分類してください。
   *'positive':感謝、理解の進化、面白い・興味深いと感じた点など、前向きな内容。
   *'negative':不満、改善要望、理解できなかった点など、後ろ向きな内容。
   *'neutral':事実の確認、質問、単なる意見など、感情が読み取れない内容。
2. **Category(カテゴリ分類)**:コメント内容が何について言及しているかを判断し、`content`, `materials`, `management`, `others`のいずれかに分類してください。
   *'content':授業で扱ったテーマ、説明の分かりやすさ、議論の内容など。
   *'materials':スライドの見やすさ、資料の量、公開のタイミングなど。
   *'management':授業のペース、時間配分、課題の量、Omnicampus, slackのツール等の使い方など。
   *'others':上記のいずれにも当てはまらない内容。
3. **Importance_score(重要度算出)**: コメントが「授業改善への貢献度」という観点でどれだけ重要かを評価し、1〜10の10段階でスコアを付けてください。以下の基準を参考にしてください。
    * **1-3 (低重要度)**: 個人的な感想や、改善に直接つながらない意見。(例: 「面白かったです」)
    * **4-7 (中重要度)**: 具体的な指摘や質問で、部分的な改善に繋がる可能性があるもの。(例: 「スライドの文字が見づらい」)
    * **8-10 (高重要度)**: 授業の根幹に関わる内容や、多くの学生に影響しうる建設的な提案・指摘。(例: 「課題の提出期限を延ばしてほしい」)
4. **Risk_level(危険度検知)**:コメント内容が何に言及しているのかを理解し、コメントの緊急性や危険性に応じて'high（緊急・危険）','medium（やや危険）','low（通常）'のいずれかに分類してください。
   *'high（緊急・危険）':講師への誹謗中傷、講義内容の重大な誤りの指摘など、**即時対応が必要な**内容。
   *'medium（やや緊急・やや危険）':講師や運営への強い批判など、**早期の確認が望ましい**内容。
   *'low（通常）':上記以外の通常コメント
5. **summary**: コメントの要約（20文字以内）
6. **keywords**: コメント内容の重要なキーワード（最大5個の配列）

# コメント例
- コメント: 「今日のTransformerの解説が非常に分かりやすかったです。特にAttentionの事例が興味深かったです。」
  - 分析結果: {{"sentiment": "positive", "category": "講義内容", "importance_score": 3, "risk_level": "low", "summary": "Transformerの解説が分かりやすかった。", "keywords": ["Transformer", "Attention", "解説", "分かりやすかった", "興味深い"]}}
- コメント: 「スライドの文字が小さくて少し見づらい箇所がありました。修正いただけると助かります。」
  - 分析結果: {{"sentiment": "negative", "category": "講義資料", "importance_score": 6, "risk_level": "low", "summary": "スライドの文字が小さく見づらい。", "keywords": ["スライド", "文字", "小さい", "見づらい"]}}
- コメント: 「課題2の提出は、システムトラブルも考慮して期限を2日ほど延ばしていただけないでしょうか。」
  - 分析結果: {{"sentiment": "negative", "category": "運営", "importance_score": 8, "risk_level": "low", "summary": "課題2の提出期限延長の要望。", "keywords": ["課題", "提出期限", "延長", "要望"]}}
- コメント: 「来週の授業はハイブリッド形式ですか？教室参加の場合、予約は必要でしょうか？」
  - 分析結果: {{"sentiment": "neutral", "category": "運営", "importance_score": 5, "risk_level": "low", "summary": "来週の授業はハイブリッド形式？", "keywords": ["来週", "授業", "ハイブリッド形式", "予約"]}}
- コメント: 「今日の講義の説明は全く意味が分からなかった。教える気がないならやめてほしい。」
  - 分析結果: {{"sentiment": "negative", "category": "講義内容", "importance_score": 7, "risk_level": "medium", "summary": "講義が全く理解できず、強い不満。", "keywords": ["説明", "意味不明", "批判", "教える気"]}}

# 分析対象コメント
コメント: "{comment}"
分析結果:
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
        
    def _get_final_importance_score(self, category: str, items: List[Dict]) -> int:
        """
        [フェーズ２] グループ化されたコメントの共通性を評価し、最終的な重要度を算出する
        """
        if not items:
            return 1

        comment_summaries = "\n".join([f"- 「{item['summary']}」 (暫定重要度: {item['importance_score']})" for item in items])
        
        prompt = f"""
あなたは大学の授業評価を分析するデータアナリストです。
現在、「{category}」に関するコメントを分析しています。
以下の通り、{len(items)}件の類似したコメントが寄せられました。

# 類似コメントの要約リスト
{comment_summaries}

# あなたのタスク
これらのコメントの**数（共通性）**と、それぞれの**内容の重要性**を総合的に判断し、このトピックに対する「最終的な重要度（Final Importance Score）」を1から10の数値で1つだけ評価してください。
特に、同じような意見が多数寄せられている場合は、重要度を高く評価してください。

あなたの回答は、評価した【数値のみ】としてください。

例: 8
"""
        try:
            response = self.model.generate_content(prompt)
            # 応答から数値を抽出し、整数に変換。失敗した場合は暫定スコアの平均を返す
            final_score = int(response.text.strip())
            return final_score
        except (ValueError, TypeError) as e:
            print(f"\n最終スコアの評価エラー ({category}): {e}。暫定スコアの平均を使用します。")
            # フォールバックとして、グループ内の暫定スコアの平均値を返す
            avg_score = sum(item['importance_score'] for item in items) / len(items)
            return round(avg_score)

    def analyze_comments_batch(self, comments: List[str], delay: float = 0.5) -> List[Dict[str, Any]]:
        """
        [オーケストレーター] 複数のコメントを二段階分析アプローチで一括分析
        """
        total = len(comments)
        if total == 0:
            return []

        # === フェーズ1: 個別分析 ===
        print("フェーズ1: 各コメントを個別に分析中...")
        initial_results = []
        start_time = time.time()
        for i, comment in enumerate(comments):
            if i > 0: time.sleep(delay)
            
            result = self._analyze_single_comment_phase1(comment)
            result['original_comment'] = comment
            result['index'] = i # 元の順序を保持するためのインデックス
            initial_results.append(result)
            
            # (元の進捗表示ロジックはそのまま流用)
            progress = (i + 1) / total
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r[{bar}] {i + 1}/{total} ({progress*100:.1f}%)", end="", flush=True)
        print("\nフェーズ1完了。")

        # === フェーズ2: 共通性の評価と重要度の再計算 ===
        print("フェーズ2: コメントの共通性を評価し、重要度を再計算中...")
        
        # カテゴリに基づいてコメントをグルーピング
        grouped_by_category = defaultdict(list)
        for result in initial_results:
            grouped_by_category[result['category']].append(result)
            
        final_results_dict = {res['index']: res for res in initial_results}

        # グループごとに共通性評価を実行
        group_count = len(grouped_by_category)
        for i, (category, items) in enumerate(grouped_by_category.items()):
            # コメントが2件未満のグループは共通性評価の対象外
            if len(items) < 2:
                continue

            print(f" - カテゴリ「{category}」({len(items)}件)の共通性を評価中... ({i+1}/{group_count})")
            
            # 共通性を考慮した最終スコアを取得
            final_score = self._get_final_importance_score(category, items)
            
            # グループ内の全コメントの importance_score を最終スコアで更新
            for item in items:
                final_results_dict[item['index']]['importance_score'] = final_score
        
        print("フェーズ2完了。")
        
        # 元の順序に戻してリストとして返す
        final_results_list = [final_results_dict[i] for i in range(total)]
        return final_results_list
    
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
        sentiment_counts = defaultdict(int)
        for result in analysis_results:
            sentiment_counts[result.get("sentiment", "neutral")] += 1
        
        # カテゴリ集計
        CATEGORY_MAP = {
            "content": "講義内容",
            "materials": "講義資料",
            "management": "運営",
            "others": "その他"
        }
        category_counts_jp = defaultdict(int)
        for result in analysis_results:
            eng_category = result.get("category", "others")
            # マッピングを元に日本語カテゴリに変換してカウント
            jp_category = CATEGORY_MAP.get(eng_category, "その他")
            category_counts_jp[jp_category] += 1
        
        # 重要度の高いコメント（スコア7以上）
        high_importance = [r for r in analysis_results if r.get("importance_score", 0) >= 7]
        
        # 危険度の高いコメント
        high_risk = [r for r in analysis_results if r.get("risk_level") == "high"]
        
        return {
            "total_comments": total_comments,
            "sentiment_distribution": {k: {"count": v, "percentage": v/total_comments*100} for k, v in sentiment_counts.items()},
            "category_distribution": {k: {"count": v, "percentage": v/total_comments*100} for k, v in category_counts_jp.items()},
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