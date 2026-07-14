"""
评估引擎 - LLM-as-a-Judge自动化评估 + 传统指标
三维评分：相关性、完整性、准确性（1-5分）
传统指标：Jaccard相似度、响应时间、精确匹配率
"""

import time
import json
from typing import Dict, List
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.core.rag_engine import RAGEngine
from backend.core.database import DatabaseManager
from backend.core.models_factory import create_llm

# LLM-as-a-Judge 评分Prompt
JUDGE_PROMPT_TEMPLATE = """你是一个专业的RAG系统评估专家。请根据以下"问题+上下文+回答"进行评分。

问题：{question}
上下文：{context}
回答：{answer}

请从以下三个维度进行1-5分评分（5分最高），并给出简短理由：
1. 相关性（Relevance）：回答是否紧扣问题
2. 完整性（Completeness）：回答是否覆盖了问题的所有要点
3. 准确性（Accuracy）：回答中的事实是否正确

请严格按以下JSON格式输出，不要输出其他内容：
{{"relevance": 分数, "relevance_reason": "理由", "completeness": 分数, "completeness_reason": "理由", "accuracy": 分数, "accuracy_reason": "理由"}}"""


class EvalEngine:
    def __init__(self, rag_engine: RAGEngine, db_manager: DatabaseManager):
        self.rag = rag_engine
        self.db = db_manager
        self.judge_llm = None

    def _get_judge(self, config=None):
        """获取Judge LLM（懒加载）"""
        if self.judge_llm is None:
            # 使用RAG引擎的同一个LLM作为Judge
            self.judge_llm = self.rag.llm
        return self.judge_llm

    def judge_single(self, question: str, answer: str, context: str) -> Dict:
        """LLM-as-a-Judge：让大模型从三个维度打分"""
        judge = self._get_judge()
        prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT_TEMPLATE)
        chain = prompt | judge | StrOutputParser()

        try:
            raw = chain.invoke({
                "question": question,
                "context": context,
                "answer": answer
            })
            # 尝试解析JSON
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw)
            return {
                "relevance": scores.get("relevance", 3),
                "relevance_reason": scores.get("relevance_reason", ""),
                "completeness": scores.get("completeness", 3),
                "completeness_reason": scores.get("completeness_reason", ""),
                "accuracy": scores.get("accuracy", 3),
                "accuracy_reason": scores.get("accuracy_reason", ""),
                "avg_score": round(
                    (scores.get("relevance", 3) + scores.get("completeness", 3) + scores.get("accuracy", 3)) / 3, 2
                )
            }
        except Exception as e:
            return {
                "relevance": 3, "relevance_reason": f"评分解析失败: {e}",
                "completeness": 3, "completeness_reason": "",
                "accuracy": 3, "accuracy_reason": "",
                "avg_score": 3.0
            }

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Jaccard相似度：计算词级别重叠比例"""
        if not text1 or not text2:
            return 0.0
        # 中文按字符切分，英文按空格切分
        set1 = set(text1)
        set2 = set(text2)
        intersection = set1 & set2
        union = set1 | set2
        return round(len(intersection) / len(union), 4) if union else 0.0

    def keyword_hit_rate(self, answer: str, context: str) -> float:
        """关键词命中率：回答中的关键词在上下文中出现的比例"""
        if not answer or not context:
            return 0.0
        # 提取回答中的有效词（长度>1）
        answer_chars = set(c for c in answer if c.strip())
        context_chars = set(c for c in context if c.strip())
        if not answer_chars:
            return 0.0
        hits = answer_chars & context_chars
        return round(len(hits) / len(answer_chars), 4)

    def evaluate_single(self, question: str, answer: str, context: str,
                        expected: str = None, use_llm_judge: bool = True) -> Dict:
        """评估单个回答的质量"""
        result = {
            "question": question,
            "answer": answer,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }

        # 传统指标
        result["jaccard_similarity"] = self.jaccard_similarity(answer, context)
        result["keyword_hit_rate"] = self.keyword_hit_rate(answer, context)
        result["answer_length"] = len(answer)
        result["context_length"] = len(context)
        result["has_answer"] = bool(answer.strip())

        if expected:
            result["expected_answer"] = expected
            result["exact_match"] = (answer.strip() == expected.strip())
            result["jaccard_answer_expected"] = self.jaccard_similarity(answer, expected)

        # LLM-as-a-Judge评分
        if use_llm_judge:
            judge_result = self.judge_single(question, answer, context)
            result["llm_judge"] = judge_result
            result["final_score"] = judge_result["avg_score"]
        else:
            result["final_score"] = result["jaccard_similarity"]

        return result

    def evaluate_qa_pairs(self, qa_pairs: List[Dict], use_llm_judge: bool = True) -> List[Dict]:
        """评估多组问答对"""
        results = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            expected = qa.get("answer", "")

            # 检索相关文档
            docs = self.rag.retrieve(question)
            context = self.rag.format_docs(docs) if docs else ""

            # 生成回答
            start = time.time()
            actual = self.rag.generate(question, context)
            elapsed = time.time() - start

            # 评估
            eval_result = self.evaluate_single(
                question, actual, context, expected, use_llm_judge
            )
            eval_result["response_time"] = round(elapsed, 2)
            results.append(eval_result)

        return results

    def get_metrics_summary(self, results: List[Dict]) -> Dict:
        """汇总评估结果"""
        if not results:
            return {}

        total = len(results)
        avg_time = sum(r.get("response_time", 0) for r in results) / total
        avg_jaccard = sum(r.get("jaccard_similarity", 0) for r in results) / total
        avg_hit_rate = sum(r.get("keyword_hit_rate", 0) for r in results) / total
        avg_score = sum(r.get("final_score", 0) for r in results) / total
        exact_matches = sum(1 for r in results if r.get("exact_match"))

        summary = {
            "total_questions": total,
            "avg_response_time": round(avg_time, 2),
            "avg_jaccard_similarity": round(avg_jaccard, 4),
            "avg_keyword_hit_rate": round(avg_hit_rate, 4),
            "avg_final_score": round(avg_score, 2),
            "exact_match_rate": round(exact_matches / total, 2) if total > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }

        # 如果有LLM Judge结果，汇总三维分数
        judge_scores = [r.get("llm_judge", {}) for r in results if r.get("llm_judge")]
        if judge_scores:
            summary["llm_judge"] = {
                "avg_relevance": round(sum(s.get("relevance", 0) for s in judge_scores) / len(judge_scores), 2),
                "avg_completeness": round(sum(s.get("completeness", 0) for s in judge_scores) / len(judge_scores), 2),
                "avg_accuracy": round(sum(s.get("accuracy", 0) for s in judge_scores) / len(judge_scores), 2),
            }

        return summary
