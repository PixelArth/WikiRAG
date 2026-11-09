"""
Settings shared by api/, build_index.py, chat.py and run_eval.py.
Change a value here and the three scripts follow.
"""

import os
from pathlib import Path
from typing import Literal

from rag import RAG
from llm import LLM, Groq, Claude

CONFIG_DIRECTORY = Path(__file__).parent.resolve()

# LLM API TO USE
API = "Anthropic" # Supported : "Groq" (only for testing purpose, useless in production) | "Anthropic" (Uses Haiku 5.5)

# INDEXATION directory and exclusion : the demo corpus about the Solar System shipped in Demo/
MARKDOWN_DIR = CONFIG_DIRECTORY / "Demo/pages"
META_FILE = CONFIG_DIRECTORY / "Demo/pages.json"
EXCLUDE_CATEGORIES = [] # categories (see META_FILE) left out of the index, for example ["Moons"]

# Store
STORE = "In-Memory" # "In-Memory" | "FAISS"
EMBEDDING = "BGE-M3" # "mini" (d=384)| "Qwen3-0.6B" (d=1024) | "Qwen3-4B" (d=2560) | "BGE-M3" (d=1024) are good embeddings to start with
INDEX_ROOT = CONFIG_DIRECTORY / "Index/demo" # the index of an embedding model is saved in INDEX_ROOT/<embedding>

# RETRIEVAL parameters
N_DOCS = 8
HYBRID_WEIGHTS = [0.8, 0.2] # [similarity, lexical]
RETRIEVER = "hybrid" # retriever used to answer : "hybrid", "similarity" or "lexical"

# ANSWERING parameters
GENERATOR_MODEL = "openai/gpt-oss-20b" # IGNORED IF API = "Anthropic" : qwen/qwen3.8-27b | openai/gpt-oss-120b | openai/gpt-oss-20b
JUDGE_MODEL = "openai/gpt-oss-120b" # IGNORED IF API = "Anthropic" : qwen/qwen3.8-27b | openai/gpt-oss-120b | openai/gpt-oss-20b
MAX_TOKENS = 1024
EFFORT = "low" # "low" | "medium" | "high". Ignored if not an openai model
TEMPERATURE = 0.5

# Evaluation parameters
QUESTIONS_FILE = CONFIG_DIRECTORY / "Demo/eval_questions.txt" # questions, expected sources and reference answers
PAUSE_SECONDS = 2 # pause between two questions of the evaluation (API rate limits)

SYSTEM = """
    You are an assistant that helps to answer questions about the Solar System : the Sun, the planets, their moons,
    dwarf planets, asteroids, comets and the space missions that explored them.
    You can only answers to these questions only by using extracts given in-between <document> tags.

    Rules :
    - Give the document <id> for each source you USE. No Exception. 
    - For each source used, place an indicator : [i] with i the number of the source in order of appearance. List the sources at the end.
    - If no documents contain the answer, do not invent the answer and simply answer "No answer found.", nothing else. Verify the absence of the answer before answering.
    - Documents are encyclopedia pages and can include lists of figures, re-arrange them in your response to create full sentences.
    - The answer must be given in english in a formal way.
"""
JUDGE_SYSTEM = "You are a strict grader."


def indexDir(embedding : str = EMBEDDING) -> str:
    """
    Folder where the index of an embedding model is saved
    """
    
    
    
    return f"{INDEX_ROOT}/{embedding}"

def getApiKey(api : str = "Groq") -> str:
    """
    Reads the API key from the environment variable
    """
    if api == "Groq":
        key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("The environment variable GROQ_API_KEY is not set.")
    return key

def loadRag() -> RAG:
    """
    Loads the saved index (no chunk is embedded again) and creates the retrievers
    """
    rag = RAG.fromIndex(indexDir(), vector_store=STORE)
    rag.create_retriever(similarity=True, lexical=True, n_document_retrieved=N_DOCS, hybrid_weights=HYBRID_WEIGHTS)
    return rag

def createGenerator(api : Literal["Groq","Anthropic"] | None = "Groq") -> LLM:
    """
    LLM that answers the questions from the retrieved documents
    """
    if api == "Groq":
        return Groq(key=getApiKey(), system_prompt=SYSTEM, model=GENERATOR_MODEL,
                temperature=TEMPERATURE, max_token=MAX_TOKENS, effort=EFFORT)
    elif api == "Anthropic":
        return Claude(key=None, system_prompt=SYSTEM, model="claude-haiku-5-5", max_token=MAX_TOKENS, effort=EFFORT, thinking=True)

def createJudge(api : Literal["Groq","Anthropic"] | None = "Groq") -> Groq:
    """
    LLM that grades the answers of the generator : no randomness, enough tokens to write its JSON verdict
    """
    if api == "Groq":
        return Groq(key=getApiKey(), system_prompt=JUDGE_SYSTEM, model=JUDGE_MODEL,
                    temperature=0.0, max_token=1024, effort="low")
    elif api == "Anthropic":
        return Claude(key=None, system_prompt=JUDGE_SYSTEM, model="claude-haiku-5-5", max_token=1024, effort='medium', thinking=True)
    
