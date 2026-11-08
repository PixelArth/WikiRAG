"""
ANSWERING : asks questions about the indexed documents in the terminal : python chat.py
Needs the index built by build_index.py.
"""

import config
from pipeline import RAGPipeline

def main() -> None:
    ragpipeline = RAGPipeline(rag = config.loadRag(), llm = config.createGenerator(api = config.API), retriever=config.RETRIEVER)
    while True:
        query = input("* -> Ask anything about the documents !\n* -> Type 'exit' to leave.\n")
        if query == "exit":
            break
        answer = ragpipeline.QAPipeline(query=query)
        print(answer)
    print("Bye !")

if __name__ == "__main__":
    main()
