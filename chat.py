"""
ANSWERING : asks questions about BG3 in the terminal : python chat.py
Needs the index built by build_index.py.
"""

import config


def main() -> None:
    rag = config.loadRag()
    model = config.createGenerator()

    while True:
        query = input("* -> Ask anything about BG3 !\n* -> Type 'exit' to leave.\n")
        if query == "exit":
            break
        documents = rag.invoke(query=query, retriever_to_use=[config.RETRIEVER])[config.RETRIEVER]
        answer = rag.call_LLM(llm=model, query=query, documents=documents)
        print(answer)
    print("Bye !")


if __name__ == "__main__":
    main()
