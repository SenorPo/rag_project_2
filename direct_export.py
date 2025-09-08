# direct_export.py - Bypass LlamaIndex loading issues
import json
import os
from pathlib import Path


def direct_export_from_storage():
    """Export directly from storage files without loading the full index"""

    storage_path = Path("./storage")

    # Check if storage exists
    if not storage_path.exists():
        print("No storage folder found. Please build the index first.")
        return False

    documents_data = []

    try:
        # Try to read from docstore.json directly
        docstore_path = storage_path / "index" / "docstore.json"

        if docstore_path.exists():
            print("Reading from docstore.json...")
            with open(docstore_path, 'r', encoding='utf-8') as f:
                docstore_data = json.load(f)

            # Extract documents from docstore
            if 'docstore/data' in docstore_data:
                docs = docstore_data['docstore/data']
                for doc_id, doc_info in docs.items():
                    if isinstance(doc_info, dict) and 'text' in doc_info:
                        documents_data.append({
                            "id": doc_id,
                            "text": doc_info['text'],
                            "metadata": doc_info.get('metadata', {})
                        })

        # Fallback: read from documents folder
        if not documents_data:
            print("Reading from documents folder...")
            docs_path = storage_path / "documents"

            if docs_path.exists():
                for file_path in docs_path.glob("*.pdf"):
                    # For PDF files, we'll just note them (actual text extraction would need PyPDF2)
                    documents_data.append({
                        "id": file_path.stem,
                        "text": f"PDF Document: {file_path.name} (Please process through admin interface for full text)",
                        "metadata": {"filename": file_path.name, "type": "pdf"}
                    })

        if not documents_data:
            print("No documents found in storage. Please upload and process documents first.")
            return False

        # Create export data
        export_data = {
            "documents": documents_data,
            "total_docs": len(documents_data),
            "export_version": "1.0-direct",
            "note": "Exported directly from storage files"
        }

        # Save to JSON
        output_file = "vector_database.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        file_size = os.path.getsize(output_file) / 1024

        print(f"Export successful!")
        print(f"Documents exported: {len(documents_data)}")
        print(f"File size: {file_size:.1f} KB")
        print(f"File created: {output_file}")

        return True

    except Exception as e:
        print(f"Error during direct export: {e}")
        return False


if __name__ == "__main__":
    direct_export_from_storage()