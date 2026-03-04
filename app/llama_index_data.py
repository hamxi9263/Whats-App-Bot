"""
llama_prepare_data.py

ONLY prepares structured Node data using LlamaIndex.
No embeddings.
No index.
Just structured nodes preview (like old markdown file).
"""

import re
import os
import json
from datetime import datetime

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PDF_PATH = "app/data/hair_transplant.pdf"
OUTPUT_JSON = "app/data/hair_transplant_nodes.json"


# ─────────────────────────────────────────────
# STEP 1: Load PDF
# ─────────────────────────────────────────────

def load_documents(pdf_path: str):
    print("[LlamaIndex] Loading PDF...")
    documents = SimpleDirectoryReader(
        input_files=[pdf_path]
    ).load_data()
    return documents


# ─────────────────────────────────────────────
# STEP 2: Split into Nodes
# ─────────────────────────────────────────────

def parse_nodes(documents):
    print("[LlamaIndex] Splitting into nodes...")

    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )

    nodes = parser.get_nodes_from_documents(documents)
    return nodes


# ─────────────────────────────────────────────
# STEP 3: Metadata Enrichment
# ─────────────────────────────────────────────

def enrich_metadata(nodes):
    print("[LlamaIndex] Injecting structured metadata...")

    for node in nodes:
        text = node.text

        # DOCTOR TAGGING
        doctor_match = re.search(
            r'(Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)', text
        )
        if doctor_match:
            node.metadata["DATA_TYPE"] = "DOCTOR_PROFILE"
            node.metadata["DOCTOR"] = doctor_match.group(1)

        # STAGE → GRAFT
        stage_match = re.search(
            r'Stage\s*(\d+).*?(\d{3,4})\s*-\s*(\d{3,4})',
            text
        )
        if stage_match:
            node.metadata["DATA_TYPE"] = "GRAFT_REQUIREMENT"
            node.metadata["STAGE"] = stage_match.group(1)
            node.metadata["GRAFT_MIN"] = stage_match.group(2)
            node.metadata["GRAFT_MAX"] = stage_match.group(3)

        # Currency
        if "PKR" in text:
            node.metadata["CURRENCY"] = "PKR"

        if "$" in text or "USD" in text:
            node.metadata["CURRENCY"] = "USD"

        node.metadata["processed_at"] = datetime.utcnow().isoformat()

    return nodes


# ─────────────────────────────────────────────
# STEP 4: Save Prepared Nodes
# ─────────────────────────────────────────────

def save_nodes(nodes, output_path):
    print(f"[LlamaIndex] Saving structured nodes to {output_path}")

    structured_data = []

    for i, node in enumerate(nodes):
        structured_data.append({
            "node_id": i,
            "text": node.text,
            "metadata": node.metadata
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print("[LlamaIndex] Saved successfully.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    documents = load_documents(PDF_PATH)
    nodes = parse_nodes(documents)
    nodes = enrich_metadata(nodes)

    save_nodes(nodes, OUTPUT_JSON)

    print("\n" + "=" * 70)
    print("NODE PREVIEW (first 3 nodes)")
    print("=" * 70)

    for i in range(min(3, len(nodes))):
        print(f"\nNode {i}")
        print("-" * 40)
        print("Text:")
        print(nodes[i].text[:500])
        print("\nMetadata:")
        print(nodes[i].metadata)

    print("\nTotal Nodes:", len(nodes))