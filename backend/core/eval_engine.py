"""
SmartKB RAG效果评估模块

本模块实现基于LLM-as-a-Judge的RAG检索质量自动化评估。
什么是LLM-as-a-Judge？
    使用大语言模型作为"裁判"，自动评估检索结果的质量。
    类似于让AI给AI打分，比传统的关键词匹配更智能。

评估维度：
┌──────────────┬─────────────────────────────────────────┐
│    维度      │              说明                      │
├──────────────┼─────────────────────────────────────────┤
│  相关性     │ 检索结果是否与问题相关                   │
│  完整性     │ 检索结果是否包含回答所需信息             │
│  准确性     │ 检索结果是否准确无误                    │
│ 关键词命中率│ 标准答案关键词在检索结果中的命中比率     │
└──────────────┴─────────────────────────────────────────┘

使用场景：
- 评估不同分块策略（chunk_size、overlap）的效果
- 模型更换后的效果对比
- 检索参数（top_k）调优

使用示例：
    eval_engine = EvalEngine(rag_engine)
    report = eval_engine.run_evaluation()
    eval_engine.print_report(report)
"""

import json
import os
import re
from typing import List, Dict, Any
from datetime import datetime

from backend.core.rag_engine import RAGEngine


# ============================================================================
# 常量定义
# ============================================================================

# 默认测试集路径
DEFAULT_TEST_SET_PATH = "data/eval_test_set.json"

# LLM评分Prompt模板
JUDGE_PROMPT_TEMPLATE = """你是一个专业的RAG系统评估专家。请评估以下检索结果与问题的相关性。
问题：{question}

检索到的文档片段：
{retrieved_texts}

请从以下维度打分（1-5分）：
1. 相关性：检索结果是否与问题相关
2. 完整性：检索结果是否包含回答问题所需的信息
3. 准确性：检索结果是否准确无误
请直接返回JSON格式：{{"relevance": 分数, "completeness": 分数, "accuracy": 分数, "reason": "评分理由"}}

注意：只返回JSON，不要其他内容。"""


def get_default_test_set() -> List[Dict[str, str]]:
    """
    获取默认测试集

    包含10个典型企业场景测试问题。

    Returns:
        测试用例列表，每个用例包含：
        - id: 唯一标识
        - question: 测试问题
        - expected_keywords: 期望出现的关键词
        - category: 问题类别
    """
    return [
        {"id": 1, "question": "公司报销流程是什么？",
         "expected_keywords": ["报销", "流程", "审批", "提交"], "category": "政策查询"},
        {"id": 2, "question": "2024年电子产品的总销售额是多少？",
         "expected_keywords": ["电子产品", "销售额", "2024"], "category": "数据查询"},
        {"id": 3, "question": "如何申请年假？",
         "expected_keywords": ["年假", "申请", "天数", "条件"], "category": "政策查询"},
        {"id": 4, "question": "公司的组织架构是怎样的？",
         "expected_keywords": ["部门", "架构", "领导"], "category": "组织信息"},
        {"id": 5, "question": "新员工入职需要准备什么材料？",
         "expected_keywords": ["入职", "材料", "证件", "合同"], "category": "人事政策"},
        {"id": 6, "question": "办公用品如何申领？",
         "expected_keywords": ["办公用品", "申领", "采购"], "category": "行政流程"},
        {"id": 7, "question": "公司的绩效考核周期是多久？",
         "expected_keywords": ["绩效", "考核", "周期", "季度"], "category": "人事政策"},
        {"id": 8, "question": "会议室如何预约？",
         "expected_keywords": ["会议室", "预约", "系统"], "category": "行政流程"},
        {"id": 9, "question": "加班补贴政策是什么？",
         "expected_keywords": ["加班", "补贴", "调休", "规定"], "category": "政策查询"},
        {"id": 10, "question": "员工培训有哪些类型？",
         "expected_keywords": ["培训", "类型", "课程", "发展"], "category": "人事政策"}
    ]


class EvalEngine:
    """
    RAG效果评估引擎

    核心功能：
    1. 加载/管理测试集
    2. 执行检索并评估质量
    3. 生成评估报告

    Attributes:
        rag_engine: RAG引擎实例
        judge_llm: 用于打分的LLM
    """

    def __init__(self, rag_engine: RAGEngine):
        """
        初始化评估引擎

        Args:
            rag_engine: RAG引擎实例
        """
        self.rag_engine = rag_engine
        self.judge_llm = rag_engine.llm    # 复用RAG引擎的LLM作为裁判

    # ========================================================================
    # 测试集管理
    # ========================================================================

    def load_test_set(self, test_set_path: str = None) -> List[Dict]:
        """
        加载测试集

        优先从文件加载，文件不存在则使用默认测试集。

        Args:
            test_set_path: 测试集JSON文件路径

        Returns:
            测试用例列表
        """
        if test_set_path and os.path.exists(test_set_path):
            with open(test_set_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return get_default_test_set()

    def save_test_set(self, test_set: List[Dict], test_set_path: str = None):
        """
        保存测试集到文件

        Args:
            test_set: 测试用例列表
            test_set_path: 保存路径
        """
        if test_set_path is None:
            test_set_path = DEFAULT_TEST_SET_PATH

        os.makedirs(os.path.dirname(test_set_path), exist_ok=True)
        with open(test_set_path, 'w', encoding='utf-8') as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)

    # ========================================================================
    # 评估方法
    # ========================================================================

    def evaluate_single(self, test_case: Dict, top_k: int = 3) -> Dict[str, Any]:
        """
        评估单个测试用例

        评估流程：
        ┌─────────────┐
        │ 执行检索    │ (使用RAG引擎检索)
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │计算关键词   │ (传统评估方法)
        │  命中率     │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ LLM打分     │ (智能评估方法)
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ 综合评判    │ (加权平均)
        └─────────────┘

        Args:
            test_case: 测试用例
            top_k: 检索返回的文档数量

        Returns:
            评估结果字典
        """
        question = test_case["question"]
        expected_keywords = test_case.get("expected_keywords", [])

        # ---- 步骤1: 执行检索 ----
        search_results = self.rag_engine.retrieve(question, top_k=top_k)
        retrieved_texts = [r["content"] for r in search_results]
        retrieved_combined = "\n".join(retrieved_texts)

        # ---- 步骤2: 计算关键词命中率 ----
        keyword_hits = sum(1 for kw in expected_keywords if kw in retrieved_combined)
        keyword_hit_rate = keyword_hits / len(expected_keywords) if expected_keywords else 0

        # ---- 步骤3: LLM打分 ----
        scores, avg_score = self._llm_judge(question, retrieved_combined)

        return {
            "id": test_case.get("id", 0),
            "question": question,
            "category": test_case.get("category", "未分类"),
            "keyword_hit_rate": round(keyword_hit_rate, 2),
            "keyword_hits": f"{keyword_hits}/{len(expected_keywords)}",
            "llm_scores": scores,
            "avg_score": round(avg_score, 2),
            "retrieved_count": len(search_results),
            "passed": avg_score >= 3.0 and keyword_hit_rate >= 0.5
        }

    def _llm_judge(self, question: str, retrieved_texts: str) -> tuple:
        """
        使用LLM作为裁判打分

        Args:
            question: 用户问题
            retrieved_texts: 检索到的文本

        Returns:
            (scores_dict, avg_score) 元组
        """
        # 默认分数（用于异常情况）
        default_scores = {"relevance": 3, "completeness": 3, "accuracy": 3}
        default_avg = 3.0

        try:
            # 构建评估提示词
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=question,
                retrieved_texts=retrieved_texts
            )

            # 调用LLM
            response = self.judge_llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # 解析JSON评分
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                scores = json.loads(json_match.group())
                avg_score = (
                    scores.get("relevance", 3) +
                    scores.get("completeness", 3) +
                    scores.get("accuracy", 3)
                ) / 3
                return scores, avg_score

            return default_scores, default_avg

        except Exception as e:
            default_scores["error"] = str(e)
            return default_scores, default_avg

    def run_evaluation(self, test_set_path: str = None, top_k: int = 3) -> Dict[str, Any]:
        """
        运行完整评估

        执行所有测试用例，生成评估报告

        Args:
            test_set_path: 测试集文件路径
            top_k: 检索返回的文档数量

        Returns:
            评估报告字典
        """
        # 加载测试集
        test_set = self.load_test_set(test_set_path)

        # 逐个评估
        results = []
        for i, test_case in enumerate(test_set, 1):
            print(f"[评估] {i}/{len(test_set)}: {test_case['question']}")
            result = self.evaluate_single(test_case, top_k)
            results.append(result)

        # 计算汇总指标
        report = self._generate_report(results)

        return report

    def _generate_report(self, results: List[Dict]) -> Dict[str, Any]:
        """
        生成评估报告

        Args:
            results: 所有测试用例的评估结果

        Returns:
            完整的评估报告
        """
        total_count = len(results)
        passed_count = sum(1 for r in results if r["passed"])

        # 计算平均值
        avg_keyword_hit = sum(r["keyword_hit_rate"] for r in results) / total_count
        avg_llm_score = sum(r["avg_score"] for r in results) / total_count

        # 按类别统计
        category_stats = self._calculate_category_stats(results)

        return {
            "summary": {
                "total_cases": total_count,
                "passed_count": passed_count,
                "pass_rate": round(passed_count / total_count, 2),
                "avg_keyword_hit_rate": round(avg_keyword_hit, 2),
                "avg_llm_score": round(avg_llm_score, 2),
                "evaluation_time": datetime.now().isoformat()
            },
            "category_stats": category_stats,
            "details": results
        }

    def _calculate_category_stats(self, results: List[Dict]) -> Dict:
        """
        按类别统计评估结果

        Args:
            results: 评估结果列表

        Returns:
            类别统计字典
        """
        stats = {}

        for r in results:
            category = r["category"]
            if category not in stats:
                stats[category] = {"count": 0, "passed": 0, "avg_score": 0}

            stats[category]["count"] += 1
            if r["passed"]:
                stats[category]["passed"] += 1
            stats[category]["avg_score"] += r["avg_score"]

        # 计算平均分和通过率
        for category in stats:
            count = stats[category]["count"]
            stats[category]["avg_score"] = round(stats[category]["avg_score"] / count, 2)
            stats[category]["pass_rate"] = round(stats[category]["passed"] / count, 2)

        return stats

    # ========================================================================
    # 报告输出
    # ========================================================================

    def save_report(self, report: Dict, report_path: str = None) -> str:
        """
        保存评估报告到文件

        Args:
            report: 评估报告
            report_path: 保存路径（可选，默认按时间戳生成）

        Returns:
            报告文件路径
        """
        if report_path is None:
            os.makedirs("data/eval_reports", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"data/eval_reports/eval_report_{timestamp}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[评估] 报告已保存: {report_path}")
        return report_path

    def print_report(self, report: Dict):
        """
        打印格式化的评估报告

        Args:
            report: 评估报告
        """
        summary = report["summary"]

        print("\n" + "=" * 60)
        print("           RAG效果评估报告")
        print("=" * 60)
        print(f"评估时间: {summary['evaluation_time']}")
        print(f"测试用例: {summary['total_cases']}")
        print(f"通过数量: {summary['passed_count']}")
        print(f"通过率: {summary['pass_rate'] * 100:.1f}%")
        print(f"平均关键词命中率: {summary['avg_keyword_hit_rate'] * 100:.1f}%")
        print(f"平均LLM评分: {summary['avg_llm_score']:.2f}/5.0")

        print("\n--- 按类别统计 ---")
        for category, stats in report["category_stats"].items():
            print(f"{category}: {stats['count']}条 "
                  f"通过率{stats['pass_rate'] * 100:.0f}%, "
                  f"均分{stats['avg_score']}")

        print("\n--- 详细结果 ---")
        for detail in report["details"]:
            status = "✅" if detail["passed"] else "❌"
            print(f"{status} [{detail['category']}] {detail['question']}")
            print(f"   关键词命中: {detail['keyword_hits']}, LLM评分: {detail['avg_score']}")

        print("=" * 60)


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    from backend.core.database import DatabaseManager

    # 初始化组件
    db_manager = DatabaseManager()
    rag_engine = RAGEngine(db_manager, {})
    eval_engine = EvalEngine(rag_engine)

    # 运行评估
    print("开始RAG效果评估...")
    report = eval_engine.run_evaluation()

    # 输出结果
    eval_engine.print_report(report)
    eval_engine.save_report(report)
