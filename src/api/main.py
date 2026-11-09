# Base
from typing import Annotated
from pathlib import Path

# web
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends

# own
import config
from pipeline import RAGPipeline
from .schemas import Question, Source, AskResponse

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RAGPipeline(rag = config.loadRag(), llm = config.createGenerator(api=config.API), retriever=config.RETRIEVER)
    yield

app = FastAPI(lifespan=lifespan)
app.frontend("/", directory=STATIC_DIR)

def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline
PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]



@app.get("/health")
def health(pipeline: PipelineDep):
    return {"chunks": len(pipeline.rag.documents),
            "retrievers" : {
                    "similarity" : True if pipeline.rag.vector_retriever is not None else None,
                    "lexical" : True if pipeline.rag.bm25_retriever is not None else None,
                    "hybrid" : True if pipeline.rag.hybrid_retriever is not None else None
                }
            }

@app.get("/info")
def info(pipeline: PipelineDep):
    return {
        "rag" : {
            "embedding" : {
                "embeddingName" : pipeline.rag.embeddingName,
                "dimension" : pipeline.rag.dimension
            },
            "vectorstore" : type(pipeline.rag.vectorstore).__name__,
            "retriever" : [type(pipeline.rag.vector_retriever).__name__,type(pipeline.rag.bm25_retriever).__name__, type(pipeline.rag.hybrid_retriever).__name__ ]
        },
        "llm" : {
            "modelName" : pipeline.llm.model,
            "token" : pipeline.llm.usage,
            "system_prompt" : pipeline.llm.system_prompt           
        }
    }



@app.post("/ask")
def ask(query : Question, pipeline: PipelineDep):
    """
    Ask a question to an LLM which will be answered using the retrieved documents.
    --> IMPORTANT : tokens are used during this request so use with care.
    """
    # The function is empty on purpose to avoid consumming any useless tokens : this is a simple demo project 
    # and have no pretention to be used. 
     
    # answer, documents = pipeline.QAPipeline(query.question)
    # return answer, documents
    return "Empty answer",[]
    

@app.post("/retrieve")
def retrieve(query: Question, pipeline: PipelineDep) -> list[Source]:
    """
    Simple retrieve operation over the documents, no token used during this operation.
    You can use this to simply test the retriever performance without consuming tokens.
    """
    docs = pipeline.RetrievePipeline(query.question)
    return [Source(id=d.id, text=d.page_content) for d in docs]
