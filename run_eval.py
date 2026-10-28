"""
EVALUATION : evaluates the RAG on the questions of eval_questions.txt : python run_eval.py
Needs the index built by build_index.py.
- by default only the retrieval is evaluated (no API call)
- --generate : the generator answers too, --judge : a judge LLM grades the answers (needs --generate)
- --limit N  : only the first N questions
"""

import argparse

from evaluator import RAGEvaluator
from llm import Groq
import config

QUESTIONS_FILE = "./Eval/eval_questions.txt"


def main() -> None:
    args = argparse.ArgumentParser(description="Evaluates the RAG")
    args.add_argument("--generate", action="store_true", help="also generate the answers")
    args.add_argument("--judge", action="store_true", help="grade the answers with the judge LLM (needs --generate)")
    args.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = args.parse_args()
    if args.judge and not args.generate:
        raise SystemExit("--judge needs --generate.")

    rag = config.loadRag()
    generator = config.createGenerator() if args.generate else None
    judge = Groq(key=config.getApiKey(), system_prompt=config.JUDGE_SYSTEM, model=config.JUDGE_MODEL) if args.judge else None

    evaluator = RAGEvaluator(rag=rag, generator=generator, judge=judge,
                             generationKwargs=config.generationKwargs(),
                             contextRetriever=config.RETRIEVER,
                             pauseSeconds=config.PAUSE_SECONDS if args.generate else 0.0)

    report = evaluator.run(QUESTIONS_FILE,
                           config={"embedding" : config.EMBEDDING, "k" : config.N_DOCS, "hybridWeights" : config.HYBRID_WEIGHTS},
                           limit=args.limit, generate=args.generate, judge=args.judge)
    evaluator.printSummary(report)      # tables of metrics and diagnosis
    evaluator.printFailures(report)     # failed questions : expected vs retrieved chunks
    evaluator.saveReport(report)        # ./EvalReports/eval_<date>.json


if __name__ == "__main__":
    main()
