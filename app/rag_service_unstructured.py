from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import re
import shutil

UNSTRUCTURED_DB_PATH = "vector_db_unstructured"
CHAT_FILE_PATH = "app/data/hair_chats.txt"


class UnstructuredRAGService:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if os.path.exists(UNSTRUCTURED_DB_PATH):
            self.vectordb = Chroma(
                persist_directory=UNSTRUCTURED_DB_PATH,
                embedding_function=self.embeddings
            )
        else:
            self.vectordb = self._create_vector_db()

    def _load_chat_file(self) -> str:
        loader = TextLoader(CHAT_FILE_PATH, encoding="utf-8")
        return loader.load()[0].page_content

    def _parse_into_chunks(self, raw_text: str) -> list[Document]:
        """
        Data analysis showed:
        - Short convos: 200-300 chars (1 exchange)   → keep whole
        - Medium convos: 400-600 chars (2 exchanges) → keep whole
        - Long convos: 800-1200 chars (3-4 exchanges) → split per exchange pair

        Strategy: split each conversation into (patient_question + sara_answer) pairs.
        This keeps Q&A meaning intact and avoids cutting Sara's reply mid-sentence.
        Each chunk = one complete exchange with full context.
        """
        pattern = r"(-{7} CHAT LOG \d+ \|.*?-{7})"
        parts = re.split(pattern, raw_text)

        chunks = []
        current_header = "GENERAL"
        current_metadata = {}

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if re.match(r"-{7} CHAT LOG", part):
                current_header = part
                current_metadata = self._extract_metadata(part)
                continue

            if len(part) < 80:
                continue

            lines = [l.strip() for l in part.split('\n') if l.strip()]

            # Group lines into (patient_turn, sara_turn) exchange pairs
            exchanges = []
            i = 0
            while i < len(lines):
                line = lines[i]

                # Collect patient lines (not Sara)
                patient_block = []
                while i < len(lines) and not lines[i].startswith('Sara:'):
                    patient_block.append(lines[i])
                    i += 1

                # Collect Sara's response
                sara_block = []
                while i < len(lines) and lines[i].startswith('Sara:'):
                    sara_block.append(lines[i])
                    i += 1

                if sara_block:  # only keep exchanges that have Sara's answer
                    exchanges.append({
                        "patient": "\n".join(patient_block),
                        "sara": "\n".join(sara_block)
                    })

            # If conversation has 2 or fewer exchanges — keep as one chunk
            # Data shows avg Sara reply = 250 chars, so 2 exchanges ~ 500-600 chars
            if len(exchanges) <= 2:
                content = f"{current_header}\n{part}"
                chunks.append(Document(page_content=content, metadata=current_metadata))
            else:
                # Split into individual exchange chunks for long conversations
                # Each chunk = header + one Q&A pair for precise retrieval
                for exchange in exchanges:
                    if not exchange["sara"]:
                        continue
                    content = f"{current_header}\n{exchange['patient']}\n{exchange['sara']}"
                    if len(content) > 80:
                        chunks.append(Document(page_content=content, metadata=current_metadata))

        return chunks

    def _extract_metadata(self, header: str) -> dict:
        metadata = {"source": "chat_log", "channel": "unknown", "log_id": "unknown"}
        try:
            log_match = re.search(r"CHAT LOG (\d+)", header)
            if log_match:
                metadata["log_id"] = log_match.group(1)

            channel_match = re.search(r"\|\s*(WhatsApp|Website Chat|Phone|Walk-in)\s*\|", header)
            if channel_match:
                metadata["channel"] = channel_match.group(1)
        except Exception:
            pass
        return metadata

    def _create_vector_db(self) -> Chroma:
        raw_text = self._load_chat_file()
        chunks = self._parse_into_chunks(raw_text)

        print(f"[UnstructuredRAG] Total chunks created: {len(chunks)}")

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=UNSTRUCTURED_DB_PATH
        )
        return vectordb

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        return self.vectordb.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=20,
            lambda_mult=0.6
        )

    def debug_retrieve(self, query: str, k: int = 5) -> None:
        print(f"\n{'='*60}\nDEBUG: {query}\n{'='*60}")
        for i, (doc, score) in enumerate(self.vectordb.similarity_search_with_score(query, k=k)):
            print(f"\n[{i+1}] Score: {score:.4f} | Log: {doc.metadata.get('log_id')}")
            print(doc.page_content[:250])
        print(f"{'='*60}\n")

    def rebuild_db(self) -> None:
        if os.path.exists(UNSTRUCTURED_DB_PATH):
            shutil.rmtree(UNSTRUCTURED_DB_PATH)
        self.vectordb = self._create_vector_db()