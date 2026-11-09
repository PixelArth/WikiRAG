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
        
    
    def QAPipeline(self, query : str) -> tuple[str,list]:
        """
        Simplified QA Pipeline in 2 steps : 
        - retrieves the most relevant documents found inside the store.
        - Execute a LLM Call by giving it the said documents and get the answer string
        
        returns :
            - LLM answer : str
            - documents : list of documents retrieved
        """
        documents = self.RetrievePipeline(query=query)
        return self.rag.call_LLM(llm=self.llm, query=query, documents=documents), documents
    
    def RetrievePipeline(self, query : str) -> dict:
        """
        First step of the QAPipeline with only the retrieve step.
        Retrieves the most relevant documents found inside the store.
        
        returns :
        - documents : list of documents retrieved
        """
        return self.rag.invoke(query=query, retriever_to_use=[self.retriever])[self.retriever]