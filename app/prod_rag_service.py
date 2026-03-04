from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil
import re
import hashlib

DB_PATH  = "prd_vector_db"
DOC_PATH = "app/data/hair_transplant.md"


class RAGService:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if self._db_exists_and_valid():
            print("[StructuredRAG] Loading existing vector DB...")
            self.vectordb = Chroma(
                persist_directory=DB_PATH,
                embedding_function=self.embeddings
            )
        else:
            print("[StructuredRAG] Creating optimized vector DB...")
            self.vectordb = self._create_vector_db()


    # ─────────────────────────────────────────────
    # DB VERSION CHECK (MD HASH BASED)
    # ─────────────────────────────────────────────

    def _file_hash(self, path):
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _db_exists_and_valid(self):
        if not os.path.exists(DB_PATH):
            return False

        hash_file = os.path.join(DB_PATH, "md_hash.txt")
        if not os.path.exists(hash_file):
            return False

        current_hash = self._file_hash(DOC_PATH)
        with open(hash_file, "r") as f:
            stored_hash = f.read().strip()

        return current_hash == stored_hash


    # ─────────────────────────────────────────────
    # STRUCTURE + TAG AWARE CHUNKING
    # ─────────────────────────────────────────────

    def _create_vector_db(self):

        loader = TextLoader(DOC_PATH, encoding="utf-8")
        raw_docs = loader.load()
        full_text = raw_docs[0].page_content

        # Split by SECTION
        sections = re.split(r"\n## SECTION:", full_text)

        structured_docs = []

        for section in sections:

            if not section.strip():
                continue

            section = section.strip()

            # Extract section title
            title_match = re.match(r"(.*)", section)
            section_title = title_match.group(1).strip() if title_match else "Unknown"

            # Extract metadata tags
            data_type = self._extract_tag(section, "DATA_TYPE")
            doctor    = self._extract_tag(section, "DOCTOR")
            stage     = self._extract_tag(section, "STAGE")
            currency  = self._extract_tag(section, "CURRENCY")

            # Secondary chunking (smaller, more precise embeddings)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150
            )

            sub_chunks = splitter.split_text(section)

            for chunk in sub_chunks:
                structured_docs.append(
                    Document(
                        page_content=chunk.strip(),
                        metadata={
                            "section": section_title,
                            "data_type": data_type,
                            "doctor": doctor,
                            "stage": stage,
                            "currency": currency
                        }
                    )
                )

        print(f"[StructuredRAG] Created {len(structured_docs)} optimized chunks")

        vectordb = Chroma.from_documents(
            documents=structured_docs,
            embedding=self.embeddings,
            persist_directory=DB_PATH
        )

        # Save MD hash
        with open(os.path.join(DB_PATH, "md_hash.txt"), "w") as f:
            f.write(self._file_hash(DOC_PATH))

        print("[StructuredRAG] Vector DB created successfully.")
        return vectordb


    # ─────────────────────────────────────────────
    # TAG EXTRACTOR
    # ─────────────────────────────────────────────

    def _extract_tag(self, text, tag):
        match = re.search(rf"\[{tag}:\s*(.*?)\]", text)
        return match.group(1).strip() if match else None


    # ─────────────────────────────────────────────
    # INTELLIGENT RETRIEVAL
    # ─────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5, filter_by=None):

        if filter_by:
            docs = self.vectordb.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=12,
                lambda_mult=0.6,
                filter=filter_by
            )
        else:
            docs = self.vectordb.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=12,
                lambda_mult=0.6
            )

        return docs


    # ─────────────────────────────────────────────
    # SMART RETRIEVE (AUTO FILTER BY QUERY)
    # ─────────────────────────────────────────────

    def smart_retrieve(self, query: str, k: int = 5):

        query_lower = query.lower()

        # Auto filtering rules
        if "price" in query_lower or "cost" in query_lower:
            return self.retrieve(query, k, filter_by={"data_type": "TABLE"})

        if "graft" in query_lower or "stage" in query_lower:
            return self.retrieve(query, k, filter_by={"data_type": "GRAFT_REQUIREMENT"})

        if "doctor" in query_lower or "dr." in query_lower:
            return self.retrieve(query, k, filter_by={"data_type": "DOCTOR_PROFILE"})

        return self.retrieve(query, k)


    # ─────────────────────────────────────────────
    # DEBUG TOOL
    # ─────────────────────────────────────────────

    def debug_retrieve(self, query: str, k: int = 5):

        print("\n" + "="*70)
        print(f"QUERY: {query}")
        print("="*70)

        docs = self.smart_retrieve(query, k=k)

        for i, doc in enumerate(docs):
            print(f"\n[Chunk {i+1}]")
            print(f"Section: {doc.metadata.get('section')}")
            print(f"DataType: {doc.metadata.get('data_type')}")
            print(f"Doctor: {doc.metadata.get('doctor')}")
            print(f"Stage: {doc.metadata.get('stage')}")
            print(f"Currency: {doc.metadata.get('currency')}")
            print("-" * 50)
            print(doc.page_content[:500])
            print("...")

        print("="*70)


    # ─────────────────────────────────────────────
    # FORCE REBUILD
    # ─────────────────────────────────────────────

    def rebuild_db(self):

        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            print("[StructuredRAG] Old DB removed.")

        self.vectordb = self._create_vector_db()