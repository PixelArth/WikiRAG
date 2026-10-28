"""
Settings shared by build_index.py, chat.py and run_eval.py.
Change a value here and the three scripts follow.
"""

import os

from rag import RAG
from llm import Groq

# INDEXATION
MARKDOWN_DIR = "./Scrap/BG3WikiPages"
META_FILE = "./Scrap/BG3WikiMeta/pages.json"
EXCLUDE_CATEGORIES = ["Books", "Notes"]
EMBEDDING = "BGE-M3" # "mini", "Qwen3-0.6B", "Qwen3-4B" or "BGE-M3" are good embeddings to try to start ( from huggingface.co )
INDEX_ROOT = "./Index" # the index of an embedding model is saved in INDEX_ROOT/<embedding>

# RETRIEVAL
N_DOCS = 10
HYBRID_WEIGHTS = [0.8, 0.2] # [similarity, lexical]
RETRIEVER = "hybrid" # retriever used to answer : "hybrid", "similarity" or "lexical"

# ANSWERING
GENERATOR_MODEL = "qwen/qwen3.8-27b" # qwen/qwen3.8-27b | openai/gpt-oss-120b | openai/gpt-oss-20b
JUDGE_MODEL = "openai/gpt-oss-120b" # qwen/qwen3.8-27b | openai/gpt-oss-120b | openai/gpt-oss-20b
MAX_TOKENS = 1024
EFFORT = "medium"
TEMPERATURE = 0.5
PAUSE_SECONDS = 60 # pause between two questions of the evaluation (API rate limits)

SYSTEM = """
    You are an assistant that helps to answer a lot of different questions about a video-game : Baldur's Gate III,
    which bases its gameplay on the table game Dungeons & Dragons, with quests, characters, magic and mythological creatures.
    You can only answers to these questions only by using extracts given in-between <documents> tags.

    Rules :
    - Give the document id for each source you USE. No Exception. List the sources at the end of your answer.
    - If no documents contain the answer, do not invent the answer and simply answer "No answer found.", nothing else.
    - Documents are data about a video-game and can include uncommon abreviation and table-like data, re-arrange them in your response to create full sentences.
    - The answer must be given in english in a formal way.
"""
JUDGE_SYSTEM = "You are a strict grader."


def indexDir(embedding : str = EMBEDDING) -> str:
    """
    Folder where the index of an embedding model is saved
    """
    return f"{INDEX_ROOT}/{embedding}"

def getApiKey() -> str:
    """
    Reads the Groq API key from the environment variable GROQ_API_KEY
    """
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("The environment variable GROQ_API_KEY is not set.")
    return key

def loadRag() -> RAG:
    """
    Loads the saved index (no chunk is embedded again) and creates the retrievers
    """
    rag = RAG.fromIndex(indexDir())
    rag.create_retriever(similarity=True, lexical=True, n_document_retrieved=N_DOCS, hybrid_weights=HYBRID_WEIGHTS)
    return rag

def createGenerator() -> Groq:
    """
    LLM that answers the questions from the retrieved documents
    """
    return Groq(key=getApiKey(), system_prompt=SYSTEM, model=GENERATOR_MODEL)

def generationKwargs() -> dict:
    """
    Parameters of the generation, the documents and the prompt are added by the caller
    """
    return {"temperature" : TEMPERATURE, "effort" : EFFORT, "max_token" : MAX_TOKENS}
