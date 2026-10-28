"""
INDEXATION : parses the scrapped pages, embeds the chunks and saves them in INDEX_ROOT/<embedding>.
Run it once per embedding model (or after a change in the parser) : python build_index.py
chat.py and run_eval.py load the saved files, nothing is embedded again.
"""

from document_parser import MarkdownParser
from rag import RAG
import config


def main() -> None:
    # STEP 1 : parse scrapped documents
    parser = MarkdownParser(dirPath=config.MARKDOWN_DIR,
                            metaFilePath=config.META_FILE,
                            cleanFilenames=True,
                            exclude_categories=config.EXCLUDE_CATEGORIES)
    chunks = parser.chunkAllDocs()

    # STEP 2 : embed the chunks with the desired embedding model
    rag = RAG(vector_store="In-Memory", embedding=config.EMBEDDING)
    vectors = rag.embedChunks(chunks) # numpy array with all vectors

    # STEP 3 : save chunks and vectors
    rag.saveIndex(config.indexDir(), chunks, vectors)
    print(f"Indexation done : {len(chunks)} chunks saved in {config.indexDir()}")


if __name__ == "__main__":
    main()
