from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        df = result.to_pandas()
        import math

        def is_valid_num(val):
            try:
                v = float(val)
                return not math.isnan(v)
            except:
                return False

        # If RAGAS returned NaN on all metrics, trigger heuristic fallback
        if not any(is_valid_num(result.get(m)) for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]):
            raise ValueError("RAGAS returned NaN on all metrics")

        per_question = [
            EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=float(row.get("faithfulness", 0.0) if is_valid_num(row.get("faithfulness")) else 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) if is_valid_num(row.get("answer_relevancy")) else 0.0),
                context_precision=float(row.get("context_precision", 0.0) if is_valid_num(row.get("context_precision")) else 0.0),
                context_recall=float(row.get("context_recall", 0.0) if is_valid_num(row.get("context_recall")) else 0.0),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": float(result.get("faithfulness", 0.0) if is_valid_num(result.get("faithfulness")) else 0.0),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0) if is_valid_num(result.get("answer_relevancy")) else 0.0),
            "context_precision": float(result.get("context_precision", 0.0) if is_valid_num(result.get("context_precision")) else 0.0),
            "context_recall": float(result.get("context_recall", 0.0) if is_valid_num(result.get("context_recall")) else 0.0),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed (falling back to offline evaluation): {e}")
        per_question = []
        f_scores, ar_scores, cp_scores, cr_scores = [], [], [], []
        for q, a, ctxs, gt in zip(questions, answers, contexts, ground_truths):
            q_words = set(q.lower().split())
            gt_words = set(gt.lower().split())
            a_words = set(a.lower().split())
            ctx_words = set(" ".join(ctxs).lower().split())

            cr = len(gt_words & ctx_words) / max(len(gt_words), 1)
            cp = 0.9 if any(len(gt_words & set(c.lower().split())) / max(len(gt_words), 1) > 0.3 for c in ctxs[:1]) else 0.4
            f = len(a_words & ctx_words) / max(len(a_words), 1) if a_words else 0.0
            ar = min(1.0, 0.5 + (len(a_words & q_words) / max(len(q_words), 1))) if q_words else 0.5

            cr = min(max(cr, 0.0), 1.0)
            cp = min(max(cp, 0.0), 1.0)
            f = min(max(f, 0.0), 1.0)
            ar = min(max(ar, 0.0), 1.0)

            f_scores.append(f)
            ar_scores.append(ar)
            cp_scores.append(cp)
            cr_scores.append(cr)

            per_question.append(EvalResult(
                question=q,
                answer=a,
                contexts=ctxs,
                ground_truth=gt,
                faithfulness=f,
                answer_relevancy=ar,
                context_precision=cp,
                context_recall=cr,
            ))

        avg = lambda lst: sum(lst) / max(len(lst), 1)
        return {
            "faithfulness": avg(f_scores),
            "answer_relevancy": avg(ar_scores),
            "context_precision": avg(cp_scores),
            "context_recall": avg(cr_scores),
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored_results = []
    for item in eval_results:
        metrics_dict = {
            "faithfulness": float(item.faithfulness),
            "answer_relevancy": float(item.answer_relevancy),
            "context_precision": float(item.context_precision),
            "context_recall": float(item.context_recall),
        }
        avg_score = sum(metrics_dict.values()) / 4.0
        worst_metric = min(metrics_dict, key=metrics_dict.get)
        worst_score = metrics_dict[worst_metric]
        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric, ("Unknown failure", "Review pipeline parameters")
        )

        scored_results.append({
            "avg_score": avg_score,
            "data": {
                "question": item.question,
                "worst_metric": worst_metric,
                "score": worst_score,
                "avg_score": avg_score,
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        })

    scored_results.sort(key=lambda x: x["avg_score"])
    return [item["data"] for item in scored_results[:bottom_n]]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if not dir_name and os.path.isdir("reports"):
        reports_path = os.path.join("reports", path)
        with open(reports_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
