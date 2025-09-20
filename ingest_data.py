import os
import json
import shutil
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# DATA_PATH = "recipe_corpus_test.json"
DATA_PATH = "recipe_corpus_full.json"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
BATCH_SIZE = 5


def process_and_yield_documents(file_path: str):
    """
    Generator function that reads the JSONL file line by line, processes
    each line into a LangChain Document, and yields it.
    This avoids loading the entire file into memory.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                name = record.get("name", "")
                ingredients = ", ".join(record.get("recipeIngredient", []))
                instructions = " ".join(record.get("recipeInstructions", []))
                page_content = f"{name}\n食材: {ingredients}\n制作步骤: {instructions}"
                metadata = {
                    "title": record.get("name", "No Title"),
                    "author": record.get("author", "Unknown Author"),
                    "dish_type": record.get("dish", "Unknown"),
                }
                yield Document(page_content=page_content, metadata=metadata)
            except (json.JSONDecodeError, Exception):
                # Skip malformed lines
                continue


def count_lines(file_path: str) -> int:
    """Counts the total number of lines in a file for the progress bar."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def main():
    """Main function to ingest data into the Chroma vector store."""
    print("--- Starting Data Ingestion ---")

    # Clear the existing ChromaDB directory to ensure a fresh start.
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Clearing existing database at {CHROMA_DB_PATH}...")
        shutil.rmtree(CHROMA_DB_PATH)
        print("Database cleared.")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}.")
        return

    # Initialize components
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=150)

    # Addresses the 'requested context size too large' warning by aligning the
    # client's requested context length with the model's actual context length.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, num_ctx=2048)

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

    # Get total documents for progress bar
    total_docs = count_lines(DATA_PATH)

    # Use a generator to process documents and chunk them in memory
    doc_generator = process_and_yield_documents(DATA_PATH)

    current_batch = []

    # Use tqdm for a real-time progress bar
    with tqdm(total=total_docs, desc="Processing Documents", unit="doc") as pbar:
        for doc in doc_generator:
            current_batch.append(doc)
            if len(current_batch) >= BATCH_SIZE:
                try:
                    # Split and add the batch to the vector store
                    chunks = text_splitter.split_documents(current_batch)
                    vectorstore.add_documents(chunks)
                except Exception as e:
                    print(f"\n[WARNING] Failed to process a batch. Error: {e}. Skipping this batch.")
                # Reset the batch and update progress bar
                pbar.update(len(current_batch))
                current_batch = []

        # Process the final batch if it's not empty
        if current_batch:
            try:
                chunks = text_splitter.split_documents(current_batch)
                vectorstore.add_documents(chunks)
                pbar.update(len(current_batch))
            except Exception as e:
                print(f"\n[WARNING] Failed to process the final batch. Error: {e}.")

    print("\n--- Data Ingestion Complete ---")
    print(f"Vector store now contains approximately {vectorstore._collection.count()} documents.")


if __name__ == "__main__":
    main()
