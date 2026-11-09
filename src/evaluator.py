"""
Evaluates the RAG on a set of questions with expected sources : retrieval quality, generation quality, and a trace of each run.

Usage :
    evaluator = RAGEvaluator(rag=rag, generator=model, judge=judge)
    report = evaluator.run(questions="./eval_questions.txt", config={"embedding" : "mini"})
    evaluator.printSummary(report)
    evaluator.printFailures(report)
    evaluator.saveReport(report)
"""

from typing import Callable, Literal, TypedDict
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import math
import re
import time

from langchain_core.documents import Document

from llm import LLM, LLMError
from rag import RAG


RetrieverName = Literal["hybrid", "similarity", "lexical"]

class EvalQuestion(TypedDict):
    """
    One question of the evaluation set
    - id : identifier of the question (ex : 'Q01')
    - type : kind of question (lookup, location, quest, multi-hop, trap, paraphrase, unanswerable)
    - question : text sent to the RAG
    - gold : expected sources, a page ('Fireball') or a section of a page ('Fireball#Properties'). Empty if there is no answer in the corpus
      ';' separates the sources that are all needed, '|' separates alternative sources of the same answer (any of them is enough)
    - answer : reference answer written by a human, used by the judge
    """
    id : str
    type : str
    question : str
    gold : list[str]
    answer : str


class RAGEvaluator:
    """
    Evaluates a RAG pipeline (retrievers + LLM) on a set of questions
    - retrieval : hit@k, recall@k, precision@k, nDCG@k and MRR of each retriever, computed on the ids of the retrieved chunks ('Page#Section')
    - generation : abstention, citations, and if a judge is given : context relevance, faithfulness, answer relevance and correctness
    - diagnosis : each question is labelled with the most probable origin of its failure (retrieval, ranking, generation...)
    The retrieval is evaluated apart from the generation, so that an error can be located before being fixed.
    """

    JUDGE_CRITERIA = ["context_relevance", "faithfulness", "answer_relevance", "correctness"]

    PASSING_DIAGNOSIS = ("ok", "correct_abstention", "retrieved")

    JUDGE_PROMPT = """You are a strict grader of a question-answering system that answers from a collection of documents.
You receive a QUESTION, the PASSAGES retrieved by the system, the ANSWER it produced and a REFERENCE answer written by a human expert.
Grade each criterion with an integer from 0 to 2 :
- context_relevance : 2 = the passages contain all the information needed to answer the question ; 1 = they contain only a part of it ; 0 = they do not contain it.
- faithfulness : 2 = every claim of the ANSWER is supported by the PASSAGES ; 1 = some claims are not supported ; 0 = most claims are not supported or contradict the passages. Ignore the REFERENCE for this criterion.
- answer_relevance : 2 = the ANSWER directly addresses the question ; 1 = it addresses it only partly ; 0 = it is off topic.
- correctness : the key facts are the ones needed to answer the QUESTION. 2 = the ANSWER agrees with the REFERENCE on every key fact ; 1 = it is partly correct (missing or inaccurate key facts) ; 0 = it contradicts the REFERENCE or misses the key fact. Details of the REFERENCE that the QUESTION does not ask for are not key facts : do not lower the score if the ANSWER omits them.
Do not reward length or style. Ignore the document ids written in the ANSWER.
Reply with a JSON object only, without any other text :
{{"context_relevance": 0, "faithfulness": 0, "answer_relevance": 0, "correctness": 0, "reason": "one short sentence"}}

QUESTION :
{question}

PASSAGES :
{passages}

ANSWER :
{answer}

REFERENCE :
{reference}"""

    def __init__(self,
                 rag : RAG,
                 generator : LLM,
                 judge : LLM | None = None,
                 retrievers : list[RetrieverName] | None = None,
                 contextRetriever : RetrieverName = "hybrid",
                 ks : list[int] | None = None,
                 nContext : int | None = None,
                 abstentionMarker : str = "no answer found",
                 pauseSeconds : float = 0.0,
                 maxRetries : int = 3,
                 judgeMaxChars : int | None = None):
        """
        - rag : the RAG with its retrievers already created
        - generator : LLM answering the questions (same system prompt as the real application)
        - judge : LLM grading the answers, should be different or stronger than the generator. None skips the grading. It replies a JSON so it needs enough tokens (max_token)
        The parameters of the generation (temperature, effort...) are set in the constructor of each LLM
        - retrievers : retrievers whose retrieval metrics are computed (all three by default)
        - contextRetriever : retriever whose chunks are given to the generator
        - ks : cut-offs of the retrieval metrics
        - nContext : number of chunks given to the generator, None gives all the retrieved chunks
        - abstentionMarker : text meaning that the generator refuses to answer (case insensitive)
        - pauseSeconds : pause after each LLM call, to respect the rate limits of the API
        - maxRetries : number of retries when the API fails
        - judgeMaxChars : maximum number of characters of each passage given to the judge, None gives the whole passage like the generator received it
        """
        self.rag = rag
        self.generator = generator
        self.judge = judge

        retrievers = retrievers if retrievers is not None else ["similarity", "lexical", "hybrid"]
        self.contextRetriever = contextRetriever
        self.retrievers = list(dict.fromkeys(retrievers + [contextRetriever]))
        self.ks = ks if ks is not None else [1, 3, 5, 10]
        self.nContext = nContext

        self.abstentionMarker = abstentionMarker.lower()
        self.pauseSeconds = pauseSeconds
        self.maxRetries = maxRetries
        self.judgeMaxChars = judgeMaxChars

        self.missingIdCount = 0 # chunks returned by a retriever without id

    # ================= #
    # Questions loading #
    # ================= #

    def loadQuestions(self, filepath : str) -> list[EvalQuestion]:
        """
        Reads the questions file : blocks starting with a '[Qxx]' line and the keys 'type', 'question', 'gold' (items separated by ';') and 'answer'
        Empty lines and lines starting with '#' are ignored
        """
        questions = []
        current = None
        with open(filepath, encoding='utf-8', mode='r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                header = re.fullmatch(r"\[(.+)\]", line)
                if header:
                    current = {"id" : header.group(1), "type" : "", "question" : "", "gold" : [], "answer" : ""}
                    questions.append(current)
                    continue

                if current is None:
                    raise ValueError(f"Line found before the first question header : '{line}'")
                key, _, value = line.partition(":")
                key, value = key.strip().lower(), value.strip()
                if key == "gold":
                    current["gold"] = [item.strip() for item in value.split(";") if item.strip()]
                elif key in ("type", "question", "answer"):
                    current[key] = value
                else:
                    raise ValueError(f"Unknown key '{key}' in question {current['id']}")

        for question in questions:
            if not question["question"]:
                raise ValueError(f"Question {question['id']} has no text")
        return questions

    # ============== #
    # Run evaluation #
    # ============== #

    def run(self,
            questions : list[EvalQuestion] | str,
            config : dict | None = None,
            limit : int | None = None,
            generate : bool = True,
            judge : bool = True) -> dict:
        """
        Runs the evaluation and returns the report : {"date", "config", "parameters", "summary", "results"}
        - questions : the questions or the path of the questions file
        - config : free description of the tested pipeline (embedding, chunking, k...), saved in the report to compare the runs
        - limit : evaluates only the first questions (quick test)
        - generate : False evaluates only the retrieval, which is fast and free
        - judge : False skips the grading of the answers
        """
        if isinstance(questions, str):
            questions = self.loadQuestions(questions)
        if limit is not None:
            questions = questions[:limit]

        self.checkRetrievers()
        self.missingIdCount = 0

        results = []
        for i, question in enumerate(questions, start=1):
            print(f"[{i}/{len(questions)}] {question['id']} : {question['question']}")
            results.append(self.evaluateQuestion(question=question, generate=generate, judge=judge))

        if self.missingIdCount > 0:
            print(f"Warning : {self.missingIdCount} retrieved chunks have no id, the sections of the gold items can not match them.")

        report = {
            "date" : datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "config" : config if config is not None else {},
            "parameters" : {
                "retrievers" : self.retrievers,
                "contextRetriever" : self.contextRetriever,
                "ks" : self.ks,
                "nContext" : self.nContext,
                "generator" : self.generator.describe() if generate else None,
                "judge" : self.judge.describe() if judge and generate and self.judge is not None else None,
            },
            "summary" : self.summarize(results),
            "results" : results,
        }
        return report

    def evaluateQuestion(self, question : EvalQuestion, generate : bool = True, judge : bool = True) -> dict:
        """
        Evaluates one question : retrieval metrics, then (if asked) the answer of the generator and its grading
        """
        result = {
            "id" : question["id"], "type" : question["type"], "question" : question["question"],
            "gold" : question["gold"], "reference" : question["answer"],
            "retrieved" : {}, "retrievalMetrics" : {}, "retrievalTime" : None,
            "answer" : None, "abstained" : None, "citations" : [], "generationTime" : None,
            "judge" : None, "diagnosis" : None, "error" : None,
        }

        # Retrieval
        start = time.perf_counter()
        retrieved = self.rag.invoke(query=question["question"], retriever_to_use=self.retrievers)
        result["retrievalTime"] = time.perf_counter() - start
        result["retrieved"] = {name : [self.getChunkId(doc) for doc in docs] for name, docs in retrieved.items()}
        if question["gold"]:
            result["retrievalMetrics"] = {name : self.retrievalMetrics(ids, question["gold"]) for name, ids in result["retrieved"].items()}

        # Generation
        if generate:
            docs = retrieved[self.contextRetriever][:self.nContext]
            try:
                start = time.perf_counter()
                result["answer"] = self.generateAnswer(question["question"], docs) or ""
                result["generationTime"] = time.perf_counter() - start
                result["abstained"] = self.isAbstention(result["answer"])
                result["citations"] = self.extractCitations(result["answer"], docs)

                if judge and self.judge is not None and question["gold"] and question["answer"]:
                    if result["abstained"]:
                        # Refusing to answer a question that has an answer is a wrong answer, no need to ask the judge
                        result["judge"] = {criterion : None for criterion in self.JUDGE_CRITERIA} | {"correctness" : 0.0, "reason" : "abstained"}
                    elif result["answer"].strip():
                        result["judge"] = self.judgeAnswer(question["question"], result["answer"], question["answer"], docs)
            except LLMError as e:
                result["error"] = str(e)

        result["diagnosis"] = self.diagnose(result)
        return result

    def checkRetrievers(self) -> None:
        """
        Checks that the retrievers to evaluate have been created in the RAG
        """
        attributes = {"similarity" : "vector_retriever", "lexical" : "bm25_retriever", "hybrid" : "hybrid_retriever"}
        for name in self.retrievers:
            if getattr(self.rag, attributes[name], None) is None:
                raise ValueError(f"Retriever '{name}' is not created, call rag.create_retriever() first (hybrid needs similarity=True and lexical=True).")

    # ========= #
    # Retrieval #
    # ========= #

    def getChunkId(self, doc : Document) -> str:
        """
        Gets the id of a retrieved chunk ('Page#Section'). If the retriever dropped it, falls back on the title of the page
        """
        chunkId = getattr(doc, "id", None)
        if chunkId:
            return chunkId
        self.missingIdCount += 1
        return f"{doc.metadata.get('title', 'unknown')}#unknown"

    @staticmethod
    def normalize(text : str) -> str:
        """
        Lowers the text and makes the underscores and the spaces equivalent, to compare page names
        """
        return re.sub(r"[\s_]+", " ", text).strip().lower()

    def matchGold(self, chunkId : str, goldItem : str) -> bool:
        """
        Tells if a chunk is a gold item : a page ('Fireball') matches all its chunks, a section ('Fireball#Properties') matches this chunk only
        The '-2' suffix added to the duplicated titles of a page is ignored
        A gold item can list alternative sources separated by '|' ('Lae'zel#Involvement|Free_Lae'zel#Walkthrough') : any of them gives the answer
        """
        chunk = self.normalize(chunkId)
        for alternative in goldItem.split("|"):
            gold = self.normalize(alternative)
            if "#" not in gold:
                if chunk.split("#")[0] == gold:
                    return True
            elif chunk == gold or re.fullmatch(re.escape(gold) + r"-\d+", chunk) is not None:
                return True
        return False

    def retrievalMetrics(self, retrievedIds : list[str], gold : list[str]) -> dict[str, float]:
        """
        Computes hit@k, recall@k, precision@k and nDCG@k for each k, and the reciprocal rank ('rr') of the first relevant chunk
        - a chunk is relevant if it matches at least one gold item
        - a gold item is found by the first chunk matching it, only this chunk gives a gain to the nDCG
        """
        relevant = []
        firstRank = {} # gold item -> rank (starting at 1) of the first chunk matching it
        for rank, chunkId in enumerate(retrievedIds, start=1):
            matched = [item for item in gold if self.matchGold(chunkId, item)]
            relevant.append(len(matched) > 0)
            for item in matched:
                firstRank.setdefault(item, rank)

        metrics = {}
        for k in self.ks:
            found = [item for item, rank in firstRank.items() if rank <= k]
            dcg = sum(1 / math.log2(rank + 1) for rank in firstRank.values() if rank <= k)
            idcg = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
            metrics[f"hit@{k}"] = float(len(found) > 0)
            metrics[f"recall@{k}"] = len(found) / len(gold)
            metrics[f"precision@{k}"] = sum(relevant[:k]) / k
            metrics[f"ndcg@{k}"] = min(dcg / idcg, 1.0)
        metrics["rr"] = 1 / min(firstRank.values()) if firstRank else 0.0
        return metrics

    # ========== #
    # Generation #
    # ========== #

    def buildDocuments(self, docs : list[Document]) -> list[dict]:
        """
        Converts the retrieved chunks to the format of the 'documents' argument of the LLM
        """
        return [{"id" : self.getChunkId(doc), "source" : {"type" : "text", "text" : doc.page_content}} for doc in docs]

    def generateAnswer(self, question : str, docs : list[Document]) -> str:
        """
        Asks the generator to answer the question with the given chunks
        """
        return self.callWithRetry(self.generator.call_LLM, prompt=question, documents=self.buildDocuments(docs))

    def isAbstention(self, answer : str) -> bool:
        """
        Tells if the answer is a refusal to answer ("No answer found.")
        """
        return self.abstentionMarker in answer.lower()

    def extractCitations(self, answer : str, docs : list[Document]) -> list[str]:
        """
        Gets the ids of the given chunks that are written in the answer (the system prompt asks the LLM to cite them)
        """
        answer = answer.lower()
        citations = [self.getChunkId(doc) for doc in docs if self.getChunkId(doc).lower() in answer]
        return list(dict.fromkeys(citations))

    def callWithRetry(self, function : Callable[..., str], **kwargs) -> str:
        """
        Calls an LLM, retries with an exponential backoff when the API fails (rate limit, timeout...) and pauses afterwards
        """
        for attempt in range(self.maxRetries + 1):
            try:
                result = function(**kwargs)
                time.sleep(self.pauseSeconds)
                return result
            except LLMError as e:
                if attempt == self.maxRetries:
                    raise
                wait = 5 * 2 ** attempt
                print(f"API error ({e}), retry in {wait}s...")
                time.sleep(wait)

    # ============ #
    # LLM as judge #
    # ============ #

    def judgeAnswer(self, question : str, answer : str, reference : str, docs : list[Document]) -> dict[str, float | str | None]:
        """
        Asks the judge to grade the answer on the four criteria (context relevance, faithfulness, answer relevance, correctness)
        Scores are between 0 and 1, the key 'reason' gives the explanation of the judge
        """
        passages = "\n\n".join(f"[{self.getChunkId(doc)}]\n{doc.page_content[:self.judgeMaxChars]}" for doc in docs)
        prompt = self.JUDGE_PROMPT.format(question=question, passages=passages, answer=answer, reference=reference)
        output = self.callWithRetry(self.judge.call_LLM, prompt=prompt)
        return self.parseJudgeOutput(output)

    def parseJudgeOutput(self, text : str) -> dict[str, float | str | None]:
        """
        Extracts the JSON of the judge and normalizes the scores (0 to 2) between 0 and 1. Missing or invalid scores are None
        """
        parsed = {criterion : None for criterion in self.JUDGE_CRITERIA} | {"reason" : ""}
        match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
        if match is None:
            return parsed
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return parsed
        if not isinstance(data, dict):
            return parsed

        for criterion in self.JUDGE_CRITERIA:
            value = data.get(criterion)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 2:
                parsed[criterion] = value / 2
        parsed["reason"] = str(data.get("reason", ""))
        return parsed

    # ========= #
    # Diagnosis #
    # ========= #

    def diagnose(self, result : dict) -> str:
        """
        Labels the most probable origin of a failure, checking the retrieval first and the generation after
        - retrieval_miss : a gold item is not in the retrieved chunks
        - ranking : gold items are retrieved but not all in the chunks given to the generator (nContext)
        - empty_answer, false_abstention : the generator gave nothing or refused to answer
        - hallucination : the generator answered a question that has no answer in the corpus
        - unfaithful, generation_error : the answer is not supported by the chunks, or is wrong although they were right
        - no_citation : correct answer without any source
        - retrieved : the retrieval is fine, the generation was not run
        - ok, correct_abstention : no problem found
        """
        if result["error"] is not None:
            return "error"

        gold = result["gold"]
        if not gold:
            if result["answer"] is None:
                return "retrieved"
            return "correct_abstention" if result["abstained"] else "hallucination"

        contextIds = result["retrieved"][self.contextRetriever]
        allFound = lambda ids : all(any(self.matchGold(i, item) for i in ids) for item in gold)
        if not allFound(contextIds):
            return "retrieval_miss"
        if not allFound(contextIds[:self.nContext]):
            return "ranking"

        if result["answer"] is None:
            return "retrieved"
        if not result["answer"].strip():
            return "empty_answer"
        if result["abstained"]:
            return "false_abstention"

        judge = result["judge"]
        if judge is not None:
            wrong = judge["correctness"] is not None and judge["correctness"] < 0.5
            unfaithful = judge["faithfulness"] is not None and judge["faithfulness"] < 0.5
            if unfaithful:
                return "unfaithful"
            if wrong:
                return "generation_error"
        if not result["citations"]:
            return "no_citation"
        return "ok"

    # ======= #
    # Reports #
    # ======= #

    @staticmethod
    def mean(values : list[float | None]) -> float | None:
        """
        Mean of the values that are not None (None if there is no value)
        """
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def summarize(self, results : list[dict]) -> dict:
        """
        Aggregates the results : retrieval metrics of each retriever, generation metrics, diagnosis counts, latencies and scores by type of question
        Retrieval metrics only use the questions that have gold items
        """
        answerable = [r for r in results if r["gold"]]
        generated = [r for r in results if r["answer"] is not None]
        answeredAnswerable = [r for r in generated if r["gold"]]

        retrieval = {}
        for name in self.retrievers:
            metrics = [r["retrievalMetrics"][name] for r in answerable]
            retrieval[name] = {("mrr" if key == "rr" else key) : self.mean([m[key] for m in metrics]) for key in metrics[0]} if metrics else {}

        generation = {}
        if generated:
            generation = {
                "correctAbstentionRate" : self.mean([float(r["abstained"]) for r in generated if not r["gold"]]),
                "falseAbstentionRate" : self.mean([float(r["abstained"]) for r in answeredAnswerable]),
                "citationRate" : self.mean([float(len(r["citations"]) > 0) for r in answeredAnswerable if not r["abstained"]]),
            }
            for criterion in self.JUDGE_CRITERIA:
                generation[criterion] = self.mean([r["judge"][criterion] for r in generated if r["judge"] is not None])

        kMax = max(self.ks)
        byType = {}
        for questionType in dict.fromkeys(r["type"] for r in results):
            subset = [r for r in results if r["type"] == questionType]
            contextMetrics = [r["retrievalMetrics"][self.contextRetriever] for r in subset if r["gold"]]
            byType[questionType] = {
                "n" : len(subset),
                "mrr" : self.mean([m["rr"] for m in contextMetrics]),
                f"recall@{kMax}" : self.mean([m[f"recall@{kMax}"] for m in contextMetrics]),
                "correctness" : self.mean([r["judge"]["correctness"] for r in subset if r["judge"] is not None]),
                "passRate" : self.mean([float(r["diagnosis"] in self.PASSING_DIAGNOSIS) for r in subset]),
            }

        return {
            "nQuestions" : len(results),
            "nAnswerable" : len(answerable),
            "nUnanswerable" : len(results) - len(answerable),
            "retrieval" : retrieval,
            "generation" : generation,
            "diagnosis" : dict(Counter(r["diagnosis"] for r in results)),
            "latency" : {
                "retrieval" : self.mean([r["retrievalTime"] for r in results]),
                "generation" : self.mean([r["generationTime"] for r in generated]),
            },
            "byType" : byType,
        }

    @staticmethod
    def formatValue(value : float | None) -> str:
        """
        Formats a metric with two decimals ('-' if it is not available)
        """
        return "-" if value is None else f"{value:.2f}"

    def printSummary(self, report : dict) -> None:
        """
        Prints the summary of a report
        """
        summary = report["summary"]
        fmt = self.formatValue
        print(f"\n=== Evaluation {report['date']} : {summary['nQuestions']} questions ({summary['nAnswerable']} answerable, {summary['nUnanswerable']} unanswerable) ===")
        if report["config"]:
            print(f"Config : {report['config']}")

        print("\nRetrieval")
        names = list(summary["retrieval"])
        print(f"  {'':<16}" + "".join(f"{name:>12}" for name in names))
        for metric in ["mrr"] + [f"{m}@{k}" for m in ("hit", "recall", "precision", "ndcg") for k in self.ks]:
            print(f"  {metric:<16}" + "".join(f"{fmt(summary['retrieval'][name].get(metric)):>12}" for name in names))

        if summary["generation"]:
            print("\nGeneration")
            for key, value in summary["generation"].items():
                print(f"  {key:<24}{fmt(value)}")

        print("\nDiagnosis")
        for label, count in sorted(summary["diagnosis"].items(), key=lambda item : -item[1]):
            print(f"  {label:<24}{count}")

        print("\nBy type of question")
        for questionType, metrics in summary["byType"].items():
            print(f"  {questionType:<14}" + "  ".join(f"{key}={fmt(value) if key != 'n' else value}" for key, value in metrics.items()))

        latency = summary["latency"]
        print(f"\nLatency (s) : retrieval {fmt(latency['retrieval'])}, generation {fmt(latency['generation'])}")

    def printFailures(self, report : dict, maxQuestions : int | None = None, maxAnswerChars : int | None = None) -> None:
        """
        Prints the failed questions with their diagnosis, the expected sources and the first retrieved chunks, to debug the retrieval before the LLM
        - maxQuestions : number of failed questions printed (None prints all of them)
        - maxAnswerChars : number of characters printed for each answer (None prints the whole answer)
        """
        failures = [r for r in report["results"] if r["diagnosis"] not in self.PASSING_DIAGNOSIS]
        print(f"\n{len(failures)} failed questions")
        for r in failures[:maxQuestions]:
            print(f"- {r['id']} [{r['type']}] {r['diagnosis']} : {r['question']}")
            print(f"    expected  : {', '.join(r['gold']) or '(no answer in the corpus)'}")
            print(f"    retrieved : {', '.join(r['retrieved'][self.contextRetriever][:5])}")
            if r["answer"]:
                print(f"    answer    : {r['answer'][:maxAnswerChars]}".replace("\n", "\n                "))
            if r["judge"] and r["judge"].get("reason"):
                print(f"    judge     : {r['judge']['reason']}")
            if r["error"]:
                print(f"    error     : {r['error']}")

    def saveReport(self, report : dict, folder : str = "./Eval") -> Path:
        """
        Saves the report as a JSON file named after its date, to keep a trace of each run
        """
        Path(folder).mkdir(parents=True, exist_ok=True)
        filepath = Path(folder) / f"eval_{report['date']}.json"
        with open(filepath, encoding='utf-8', mode='w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved in {filepath}")
        return filepath
