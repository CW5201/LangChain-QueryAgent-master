"""
评估引擎 - 检查RAG系统的回答质量
"""

import json
import time
from typing import Dict, List, Optional
from datetime import datetime

from backend.core.rag_engine import RAGEngine
from backend.core.database import DatabaseManager


class EvalEngine:
    def __init__(self, rag_engine: RAGEngine, db_manager: DatabaseManager):
        self.rag = rag_engine
        self.db = db_manager

    def evaluate_single(self, question: str, answer: str, context: str) -> Dict:
        """评估单个回答的质量"""
        result = {
            "question": question,
            "answer": answer,
            "context": context,
            "metrics": {},
            "timestamp": datetime.now().isoformat()
        }

        # 基础指标
        result["metrics"]["answer_length"] = len(answer)
        result["metrics"]["context_length"] = len(context)
        result["metrics"]["has_answer"] = bool(answer.strip())
        result["metrics"]["answer_context_overlap"] = self._calc_overlap(answer, context)

        return result

    def evaluate_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
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
            eval_result = self.evaluate_single(question, actual, context)
            eval_result["expected_answer"] = expected
            eval_result["response_time"] = round(elapsed, 2)
            eval_result["metrics"]["exact_match"] = self._exact_match(actual, expected)
            results.append(eval_result)

        return results

    def get_metrics_summary(self, results: List[Dict]) -> Dict:
        """汇总评估结果"""
        if not results:
            return {}

        total = len(results)
        avg_time = sum(r.get("response_time", 0) for r in results) / total
        avg_overlap = sum(r["metrics"].get("answer_context_overlap", 0) for r in results) / total
        exact_matches = sum(1 for r in results if r["metrics"].get("exact_match"))

        return {
            "total_questions": total,
            "avg_response_time": round(avg_time, 2),
            "avg_context_overlap": round(avg_overlap, 2),
            "exact_match_rate": round(exact_matches / total, 2) if total > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }

    def _calc_overlap(self, text1: str, text2: str) -> float:
        """计算两个文本的重叠比例"""
        if not text1 or not text2:
            return 0
        set1 = set(text1)
        set2 = set(text2)
        intersection = set1 & set2
        return len(intersection) / len(set1 | set2) if set1 | set2 else 0

    def _exact_match(self, actual: str, expected: str) -> bool:
        """简单判断是否完全匹配"""
        return actual.strip() == expected.strip()
