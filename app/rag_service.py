from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil

DB_PATH  = "vector_db"
DOC_PATH = "app/data/hair_transplant.docx"


class RAGService:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if os.path.exists(DB_PATH):
            print("[StructuredRAG] Loading existing vector DB...")
            self.vectordb = Chroma(
                persist_directory=DB_PATH,
                embedding_function=self.embeddings
            )
        else:
            print("[StructuredRAG] Creating new vector DB from docx...")
            self.vectordb = self._create_vector_db()

    def _create_vector_db(self):
        loader    = Docx2txtLoader(DOC_PATH)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,                          
            chunk_overlap=200,                       
            separators=["\n\n", "\n", ". ", " "], 
            length_function=len,
        )

        splits = text_splitter.split_documents(documents)

        print(f"[StructuredRAG] Total chunks created: {len(splits)}")

        # ── Log chunk sizes for verification ──────────────────────────────────
        sizes = [len(s.page_content) for s in splits]
        print(f"[StructuredRAG] Avg chunk size : {sum(sizes)//len(sizes)} chars")
        print(f"[StructuredRAG] Min chunk size : {min(sizes)} chars")
        print(f"[StructuredRAG] Max chunk size : {max(sizes)} chars")

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=DB_PATH
        )

        print("[StructuredRAG] Vector DB created successfully.")
        return vectordb

    def retrieve(self, query: str, k: int = 5) -> list:
        docs = self.vectordb.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=15,       # ✅ fetch wider pool first
            lambda_mult=0.6   # ✅ balance relevance + diversity
        )
        return docs


    def debug_retrieve(self, query: str, k: int = 5) -> None:
        print(f"\n{'='*65}")
        print(f"DEBUG QUERY: {query}")
        print(f"{'='*65}")

        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)

        for i, (doc, score) in enumerate(docs_and_scores):
            print(f"\n[Chunk {i+1}]")
            print(f"  Score   : {score:.4f}  {'✅ Good' if score < 0.8 else '⚠️ Weak'}")
            print(f"  Length  : {len(doc.page_content)} chars")
            print(f"  Content : {doc.page_content[:300]}")
            print(f"  {'...' if len(doc.page_content) > 300 else ''}")

        print(f"\n{'='*65}")
        print(f"SCORE GUIDE: < 0.5 Excellent | 0.5-0.8 Good | > 0.8 Weak")
        print(f"{'='*65}\n")


    def rebuild_db(self) -> None:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            print("[StructuredRAG] Old vector DB deleted.")
        self.vectordb = self._create_vector_db()
        print("[StructuredRAG] Vector DB rebuilt successfully.")