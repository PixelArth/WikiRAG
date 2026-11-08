"""
Pipeline class centralizing every calls to the RAG
"""

from typing import Literal

from llm import LLM
from rag import RAG

class RAGPipeline:
    def __init__(self, rag : RAG, llm : LLM, retriever : Literal["hybrid", "similarity", "lexical"]):
        self.rag = rag
        self.llm = llm
        self.retriever = retriever
        
    
    def QAPipeline(self, query : str) -> str:
        documents = self.RetrievePipeline(query=query)
        return self.rag.call_LLM(llm=self.llm, query=query, documents=documents)
    
    def RetrievePipeline(self, query : str) -> dict:
        return self.rag.invoke(query=query, retriever_to_use=self.retriever)[self.retriever]