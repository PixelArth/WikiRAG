import re
import json
import numpy as np
from pathlib import Path
from typing import Literal

# PyTorch
import torch

# Lanchain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# FAISS
import faiss

# Own
from llm import LLM


class PrecomputedEmbeddings(Embeddings):
    """
    Embeddings wrapper used by the vector store : the vectors of the texts already embedded (see RAG.embedChunks) are read from a table,
    any other text (a query, a new chunk) goes through the real embedding model
    """

    def __init__(self, model : Embeddings):
        self.model = model
        self.table : dict[str, np.ndarray] = {} # text -> vector

    def preload(self, texts : list[str], vectors : np.ndarray) -> None:
        """
        Registers the vectors of texts that are already embedded
        """
        self.table.update(zip(texts, vectors))

    def embed_documents(self, texts : list[str]) -> list[np.ndarray]:
        
        # Gather all texts without an associated vector
        missing = [text for text in dict.fromkeys(texts) if text not in self.table]
        
        # Creates the embedding for every texts that does not have one yet with the self.model embedding
        computed = dict(zip(missing, self.model.embed_documents(missing))) if missing else {} 
        
        # Returns the list of ALL vectors inside the table and the newly vectorized missing texts
        return [self.table[text] if text in self.table else computed[text] for text in texts]

    def embed_query(self, text : str) -> list[float]:
        return self.model.embed_query(text)


class RAG:
    # Embedding name -> Hugging Face model and size of its vectors (needed to create a FAISS index)
    EMBEDDING_MODELS = {
        "mini" : {"model_name" : "sentence-transformers/all-MiniLM-L6-v2", "dimension" : 384},
        "Qwen3-0.6B" : {"model_name" : "Qwen/Qwen3-Embedding-0.6B", "dimension" : 1024},
        "Qwen3-4B" : {"model_name" : "Qwen/Qwen3-Embedding-4B", "dimension" : 2560},
        "BGE-M3" : {"model_name" : "BAAI/bge-m3", "dimension" : 1024},
    }

    def __init__(self,
                 vector_store : Literal["In-Memory","FAISS"], 
                 embedding : Literal["mini","Qwen3-0.6B","Qwen3-4B","BGE-M3"] | None = "mini" ):
        
        print("Loading embedding model...")
        self.model = None # LLM connected to the RAG, initialiazed with the first call to LLM
        self.vector_retriever, self.bm25_retriever, self.hybrid_retriever = None,None,None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dimension = self.EMBEDDING_MODELS[embedding]["dimension"]

        huggingFaceArgs = {"model_name" : self.EMBEDDING_MODELS[embedding]["model_name"],
                           "model_kwargs" : {"device":device , "model_kwargs": {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}},
                           "encode_kwargs" : {"normalize_embeddings" : True, "batch_size": 4 if device == "cuda" else 1},
                           "show_progress" : True}
        
        
        self.embeddings = HuggingFaceEmbeddings(**huggingFaceArgs)

        print(f"Embedding model loaded on : {device}")
        self.embeddingName = embedding
        self.cachedEmbeddings = PrecomputedEmbeddings(self.embeddings) # lets the store use vectors computed beforehand


        if vector_store == "In-Memory":
            self.vectorstore = InMemoryVectorStore(embedding=self.cachedEmbeddings)
            print("Loaded in memory vectorstore")
        elif vector_store == "FAISS":
            self.vectorstore = FAISS(embedding_function = self.cachedEmbeddings,
                                     index=faiss.IndexFlatL2(self.dimension),
                                     docstore=InMemoryDocstore(),
                                     index_to_docstore_id={})
        
        print("- Embedding used : ", self.embeddingName)
        print("\tdimension : ", self.dimension)
    
    def chunkToDocs(self, chunks : list[dict]) -> list[Document]:
        """
        Turns a list of dicts containing a "text" key as a lsit of Document object from langchain_core
        """
        return [Document(page_content=chunk.pop('text'), **chunk) for chunk in chunks]

    def addChunksToStore(self, chunks : list[dict], vectors : np.ndarray | None = None) -> None:
        """
        Adds the chunks to the vector store
        - vectors : embeddings of the chunks (same order), computed by embedChunks. If None, the store embeds the chunks itself
        """
        if vectors is not None:
            if len(vectors) != len(chunks):
                raise ValueError(f"{len(vectors)} vectors for {len(chunks)} chunks.")
            if vectors.shape[1] != self.dimension:
                raise ValueError(f"Vectors of size {vectors.shape[1]}, but the embedding '{self.embeddingName}' makes vectors of size {self.dimension}.")
            self.cachedEmbeddings.preload([chunk["text"] for chunk in chunks], vectors)

        print(f"Converting text chunks to documents... ({len(chunks)} chunks)")
        docs = self.chunkToDocs(chunks)
        self.documents = docs

        print("Adding documents to store...")
        self.vectorstore.add_documents(documents=docs)

    def embedChunks(self, chunks : list[dict], blockSize : int = 1024) -> np.ndarray:
        """
        Embeds the "text" of each chunk with the embedding model, returns a float32 matrix (one row per chunk, same order)
        The texts are embedded by blocks of similar length (faster) and stored as float32 (python lists of floats would take 8 times more memory)
        """
        texts = [chunk["text"] for chunk in chunks]
        order = np.argsort([len(text) for text in texts], kind="stable")
        vectors = None
        for start in range(0, len(texts), blockSize):
            indexes = order[start:start + blockSize]
            
            #  Each elements is the vectorized version of the chunks in indexes and put in a ndarray
            block = np.asarray(self.embeddings.embed_documents([texts[i] for i in indexes]), dtype=np.float32)
            if block.shape[1] != self.dimension:
                raise ValueError(f"The model '{self.embeddingName}' makes vectors of size {block.shape[1]}, EMBEDDING_MODELS says {self.dimension}.")
            if vectors is None:
                vectors = np.empty((len(texts), block.shape[1]), dtype=np.float32)
                
            # Places the chunks at the right index in the global array to keep a consistent indexing when loading the array.
            vectors[indexes] = block
            print(f"Embedded {min(start + blockSize, len(texts))}/{len(texts)} chunks")
        return vectors

    def saveIndex(self, folder : str, chunks : list[dict], vectors : np.ndarray) -> None:
        """
        Saves an index in a folder : chunks.json (id, text, metadata), vectors.npy (embeddings) and index.json (the embedding model they come from)
        Call it before addChunksToStore, which takes the text out of the chunks
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "chunks.json", mode="w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        np.save(folder / "vectors.npy", vectors)
        with open(folder / "index.json", mode="w", encoding="utf-8") as f:
            json.dump({"embedding" : self.embeddingName, "nChunks" : len(chunks), "dimension" : int(vectors.shape[1])}, f, indent=2)

    @classmethod
    def fromIndex(cls, folder : str, vector_store : Literal["In-Memory","FAISS"] = "In-Memory") -> "RAG":
        """
        Builds a RAG from an index saved by saveIndex, with the embedding model it was made with. The chunks are not embedded again
        """
        folder = Path(folder)
        with open(folder / "index.json", encoding="utf-8") as f:
            info = json.load(f)
        with open(folder / "chunks.json", encoding="utf-8") as f:
            chunks = json.load(f)
        vectors = np.load(folder / "vectors.npy")

        rag = cls(vector_store=vector_store, embedding=info["embedding"])
        rag.addChunksToStore(chunks=chunks, vectors=vectors)
        return rag

    
    @staticmethod
    def tokenize(text : str) -> list[str]:
        """
        Tokenizer used by the BM25 retriever (documents and queries) : lowercase, no punctuation, no stop words.
        This is used for the BM25 lexical search to be more precise and focused on technical word.
        """
        stopWords = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are", "was", "do", "does",
                     "i", "my", "me", "it", "what", "which", "how", "where", "who", "can", "get", "s"}
        return [word for word in re.findall(r"\w+", text.lower()) if word not in stopWords]

    def create_retriever(self, 
                         similarity : bool = True, 
                         lexical : bool = False, 
                         n_document_retrieved : int = 10, 
                         hybrid_weights : list[float] = [0.5,0.5]) -> None:
        """
        Create all retrievers wanted to retrieve documents
        - similarity = True creates a retriever based on cosine distance bewteen vectors
        - lexical = True, creates a retriever a lexical retriever based on the BM25 algorithm, needs lanchain_community : pip install -U langchain-community rank-bm25
        if more than one retriever is used, a hybrid retriever is also created to combine results with the Reciprocal Rank Fusion (RRF) formula 
        - hybrid weights : [x,y] with x and y floats, x+y=1. x is similarity weight and y is lexical weight
        """
        self.n_document_retrieved = n_document_retrieved
        if similarity:
            self.vector_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": n_document_retrieved})
        if lexical:
            
            from langchain_community.retrievers import BM25Retriever
            from langchain_classic.retrievers import EnsembleRetriever
            # Each text is first preprocessed by RAG.tokenize before vectorization
            self.bm25_retriever = BM25Retriever.from_documents(documents=self.documents, 
                                                               k=n_document_retrieved, 
                                                               preprocess_func = self.tokenize)
        
        if  self.vector_retriever is not None and self.bm25_retriever is not None and hybrid_weights is not None :
            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights = hybrid_weights,
                c = 60)
        
    def invoke(self, 
               query : str, 
               retriever_to_use : list[Literal["hybrid", "similarity","lexical"]] | None = ["similarity"]):
        """
        Pass the user's query to the retriever
        """
        retriever_map = {"hybrid" : self.hybrid_retriever, "similarity" : self.vector_retriever, "lexical" : self.bm25_retriever}
        assert(any(val is not None for val in list(retriever_map.values()))), "No retriever found.\nRetrievers should be iniitialized through loadRag() from config.py or\n self initialized through Rag.create_retriever()"

        results = {}
        for retriever in retriever_to_use:
            # The hybrid retriever merges two lists of k documents, only the k best of the fusion are kept
            results[retriever] = retriever_map[retriever].invoke(input=query)[:self.n_document_retrieved]
        return results
        
    def call_LLM(self, 
                 llm : LLM, 
                 query: str, 
                 **kwargs):
        if self.model is None:
            self.model = llm
        documents = kwargs.get("documents")
        if documents and isinstance(documents[0], Document):
            # The LLM expects dicts {"id", "source" : {"type", "text"}}, not langchain Documents
            kwargs["documents"] = [{"id" : doc.id or doc.metadata.get("title", "unknown"), "source" : {"type" : "text", "text" : doc.page_content}} for doc in documents]
        answer = llm.call_LLM(prompt=query, **kwargs)
        return answer