# 评估引擎
# 用两种方式评估RAG回答质量：
# 1. LLM-as-a-Judge：让大模型从相关性/完整性/准确性三维打分（1-5分）
# 2. 传统指标：Jaccard相似度 + 关键词命中率

import time
import json
from typing import Dict, List
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.core.rag_engine import RAGEngine


# LLM打分的提示词
JUDGE_PROMPT = """你是RAG系统评估专家。请对以下问答进行评分。

问题：{question}
上下文：{context}
回答：{answer}

从三个维度打1-5分（5分最高），输出JSON：
{{"relevance": 分数, "completeness": 分数, "accuracy": 分数, "reason": "简短理由"}}"""


class EvalEngine:
    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine

    def judge_single(self, question, answer, context) -> Dict:
        """LLM-as-a-Judge：让大模型打分"""
        prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
        chain = prompt | self.rag.llm | StrOutputParser()
        try:
            raw = chain.invoke({"question": question, "context": context, "answer": answer})
            # 清理JSON字符串
            raw = raw.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw)
            avg = round((scores.get("relevance", 3) + scores.get("completeness", 3) + scores.get("accuracy", 3)) / 3, 2)
            return {**scores, "avg_score": avg}
        except Exception:
            return {"relevance": 3, "completeness": 3, "accuracy": 3, "avg_score": 3.0, "reason": "评分失败"}

    def jaccard(self, text1, text2) -> float:
        """Jaccard相似度：两个文本的字符重叠比例"""
        if not text1 or not text2:
            return 0.0
        s1, s2 = set(text1), set(text2)
        return round(len(s1 & s2) / len(s1 | s2), 4) if s1 | s2 else 0.0

    def keyword_hit_rate(self, answer, context) -> float:
        """关键词命中率：回答中的字符在上下文中出现的比例"""
        if not answer or not context:
            return 0.0
        answer_chars = set(c for c in answer if c.strip())
        context_chars = set(c for c in context if c.strip())
        if not answer_chars:
            return 0.0
        return round(len(answer_chars & context_chars) / len(answer_chars), 4)

    def evaluate_qa_pairs(self, qa_pairs: List[Dict], use_llm_judge=True) -> List[Dict]:
        """评估多组问答对"""
        results = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            expected = qa.get("answer", "")

            # 检索 + 生成
            docs = self.rag.retrieve(question)
            context = self.rag.format_docs(docs) if docs else ""
            start = time.time()
            actual = self.rag.generate(question, context)
            elapsed = time.time() - start

            # 计算各种指标
            result = {
                "question": question,
                "answer": actual,
                "expected_answer": expected,
                "context": context,
                "response_time": round(elapsed, 2),
                "jaccard_similarity": self.jaccard(actual, context),
                "keyword_hit_rate": self.keyword_hit_rate(actual, context),
                "exact_match": actual.strip() == expected.strip() if expected else False,
            }

            # LLM打分
            if use_llm_judge:
                result["llm_judge"] = self.judge_single(question, actual, context)
                result["final_score"] = result["llm_judge"]["avg_score"]
            else:
                result["final_score"] = result["jaccard_similarity"]

            results.append(result)
        return results

    def get_metrics_summary(self, results: List[Dict]) -> Dict:
        """汇总评估结果"""
        if not results:
            return {}
        n = len(results)
        return {
            "total_questions": n,
            "avg_response_time": round(sum(r.get("response_time", 0) for r in results) / n, 2),
            "avg_jaccard": round(sum(r.get("jaccard_similarity", 0) for r in results) / n, 4),
            "avg_hit_rate": round(sum(r.get("keyword_hit_rate", 0) for r in results) / n, 4),
            "avg_final_score": round(sum(r.get("final_score", 0) for r in results) / n, 2),
            "exact_match_rate": round(sum(1 for r in results if r.get("exact_match")) / n, 2),
            "timestamp": datetime.now().isoformat(),
        }
