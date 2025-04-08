import asyncio
import datetime
import logging
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterator

import chromadb
import nltk
import tiktoken
from chromadb.config import Settings
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.model_downloader import download_models
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.types.doc.document import TableItem
from docling_core.types.doc.labels import DocItemLabel
from nicegui import events, ui
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from unstructured.partition.auto import partition

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.debug("Starting the application")


async def async_convert(converter, source) -> ConversionResult:
    """Asynchronously convert a document using the DocumentConverter."""
    logger.debug(f"Converting {source}")
    # Use a thread pool to run the blocking convert method
    loop = asyncio.get_running_loop()
    dl_doc = await loop.run_in_executor(None, converter.convert, source)
    logger.debug(f"Finished converting {source}")
    return dl_doc


## Docling

# Download Docling models if we don't have them already
dockling_artifacts_path = Path("./.dockling_artifacts")
os.environ["DOCKLING_ARTIFACTS_PATH"] = str(dockling_artifacts_path.resolve())
logger.debug(f"Dockling artifacts path: {os.environ['DOCKLING_ARTIFACTS_PATH']}")
if not dockling_artifacts_path.exists():
    download_models(dockling_artifacts_path, progress=True)

# pipeline_options = PdfPipelineOptions(artifacts_path=dockling_artifacts_path)
# To use the locally downloaded models, use pipleline options.  Example:
# doc_converter = DocumentConverter(
#     format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
# )
# Or set the environment variable DOCKLING_ARTIFACTS_PATH to the path of the
# downloaded models

docling_pdf_pipeline_options = PdfPipelineOptions()
docling_pdf_pipeline_options.generate_picture_images = True
docling_pdf_pipeline_options.do_table_structure = True
docling_pdf_pipeline_options.table_structure_options.do_cell_matching = True

docling_format_options = {
    InputFormat.PDF: PdfFormatOption(
        pipeline_options=docling_pdf_pipeline_options,
    ),
}

## Chunking and Embedding

# ChromaDB Default is Sentence Transformers all-MiniLM-L6-v2; defining here for chunker
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
MAX_CHUNK_TOKENS = 256  # Max tokens per chunk - also the default for all-MiniLM-L6-v2

nltk.download("punkt")  # Ensure the Punkt tokenizer is available

# Initialize the ChromaDB persistent client

client = chromadb.PersistentClient(
    path="./chroma_persist",
    settings=Settings(
        allow_reset=True,
    ),
)

COLLECTION_METADATA = {
    "encoder": EMBED_MODEL_ID,
    "chunk_token_size": MAX_CHUNK_TOKENS,
    "hnsw:space": "cosine",
    "hnsw:search_ef": 100,
}


# By default, chroma will use all-MiniLM-L6-v2 sentence transformer model
collection = client.get_or_create_collection(
    name="docs",
    metadata=COLLECTION_METADATA,
)


# Store previews in memory before committing
preview_state = {"chunks": [], "doc_id": None, "source": "", "label": None}


# def unstructured_extract_text(file_path: str) -> list[str]:
#     """Extract text from a given file and return a list of strings."""
#     elements = partition(filename=file_path)
#     return [str(e) for e in elements if str(e).strip()]


async def extract_text(file_path: str, original_filename: str = None) -> list[dict]:
    """Extract text from a file and return a list of chunks."""
    document_as_chunks = []
    converter = DocumentConverter(format_options=docling_format_options)

    if not original_filename:
        original_filename = os.path.basename(file_path)

    # Generate a unique doc_id per document
    doc_id = str(uuid.uuid4())
    chunk_id = 0  # Initialize chunk_id for this document

    # Load and convert the document asynchronously
    logger.debug("Starting async conversion")
    dl_doc = await async_convert(converter, file_path)
    dl_doc = dl_doc.document
    logger.debug("Finished async conversion")

    # Chunk the document
    chunker = HybridChunker(
        tokenizer=tokenizer, max_tokens=MAX_CHUNK_TOKENS, merge_peers=True
    )
    for chunk in chunker.chunk(dl_doc):
        items = chunk.meta.doc_items
        # if len(items) == 1 and isinstance(items[0], TableItem):
        #     continue  # we will do this later
        refs = " ".join(item.get_ref().cref for item in items)
        text = chunker.serialize(chunk)
        document_chunk = {
            "text": text,
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "source": original_filename,
                "ref": refs,
            },
        }
        document_as_chunks.append(document_chunk)
        # # Handle Tables
        # for table in dl_doc.tables:
        #     if table.label in [DocItemLabel.TABLE]:
        #         ref = table.get_ref().cref
        #         print(ref)
        #         text = table.export_to_markdown()

        #         table = {
        #             "page_content": text,
        #             "metadata": {
        #                 # "doc_id": (doc_id := doc_id + 1),
        #                 "source": source,
        #                 "ref": ref,
        #             },
        #         }

        # tables.append(table)
        # documents.append(table)
        chunk_id += 1

    return document_as_chunks


async def xextract_text(file_path: str) -> list[dict]:
    """Extract text from a file and return a list of strings."""
    # TODO the path is the temp file, we need to capture the original file name
    document_as_chunks = []
    tables = []  # TODO
    doc_id = 0
    file_paths = file_path if isinstance(file_path, list) else [file_path]
    converter = DocumentConverter(format_options=docling_format_options)
    for source in file_paths:
        # Load and convert the document
        # dl_doc = converter.convert(source).document
        logger.debug("starting async")
        dl_doc = await async_convert(converter, source)
        dl_doc = dl_doc.document
        logger.debug("finished async")

        # Chunker
        chunker = HybridChunker(
            tokenizer=tokenizer, max_tokens=MAX_CHUNK_TOKENS, merge_peers=True
        )
        chunks: Iterator[BaseChunk] = chunker.chunk(dl_doc)
        for chunk in chunks:
            items = chunk.meta.doc_items
            # if len(items) == 1 and isinstance(items[0], TableItem):
            #     continue  # we will do this later

            refs = " ".join(map(lambda item: item.get_ref().cref, items))
            text = chunker.serialize(chunk)
            document_chunk = {
                "text": text,
                "metadata": {
                    "doc_id": doc_id,
                    "source": source,
                    "ref": refs,
                },
            }
            document_as_chunks.append(document_chunk)
            doc_id += 1

            # # Handle Tables
            # for table in dl_doc.tables:
            #     if table.label in [DocItemLabel.TABLE]:
            #         ref = table.get_ref().cref
            #         print(ref)
            #         text = table.export_to_markdown()

            #         table = {
            #             "page_content": text,
            #             "metadata": {
            #                 # "doc_id": (doc_id := doc_id + 1),
            #                 "source": source,
            #                 "ref": ref,
            #             },
            #         }

            # tables.append(table)
            # documents.append(table)

        # Handle images TODO
        # see https://www.ibm.com/think/tutorials/build-multimodal-rag-langchain-with-docling-granite

    return document_as_chunks
    # text = dl_doc.export_to_markdown()


def commit_chunks(chunks, label: str = None) -> tuple[int, set]:
    """Commit the chunks to the database."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    documents = [doc["text"] for doc in chunks]
    metadatas = [doc["metadata"] for doc in chunks]

    # Collect unique doc_ids from metadatas
    doc_ids = {meta["doc_id"] for meta in metadatas}

    ids = []
    # furher enrich the metadata
    for meta in metadatas:
        # Add label if provided
        if label:
            meta["label"] = label
        meta["uploaded_at"] = now
        # Optionally add encoder and chunk_token_size
        meta["encoder"] = EMBED_MODEL_ID
        meta["chunk_token_size"] = MAX_CHUNK_TOKENS
        # Generate a unique ID for each chunk
        unique_id = f"{meta['doc_id']}_chunk_{meta['chunk_id']}"
        ids.append(unique_id)

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return len(chunks), doc_ids


def clear_all_data() -> bool:
    """Clear all data in the collection."""
    return client.reset()


@ui.page("/")
def main():
    ui.label("Upload & Search").classes("text-2xl font-bold mb-4")

    label_input = ui.input("Label for this document").classes("w-full")
    preview_area = ui.column().classes("mt-2 p-2 rounded")

    async def handle_upload(upload_event: events.UploadEventArguments, callback):
        original_filename = upload_event.name
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(original_filename)[-1]
        ) as tmp:
            tmp.write(upload_event.content.read())
            tmp_path = tmp.name

        documents = await extract_text(tmp_path, original_filename=original_filename)
        try:
            os.remove(tmp_path)
        except Exception as e:
            logger.debug(f"Error removing temporary file: {e}")

        callback(documents)

    def load_preview(chunks):
        # Store preview info
        preview_state["chunks"] = chunks
        # Extract doc_id and filename from the chunks
        doc_ids = {chunk["metadata"]["doc_id"] for chunk in chunks}
        sources = {chunk["metadata"]["source"] for chunk in chunks}
        preview_state["doc_ids"] = doc_ids
        preview_state["label"] = label_input.value
        # Assuming only one source per upload
        source = sources.pop() if sources else ""
        preview_state["source"] = source
        preview_area.clear()
        with preview_area:
            ui.label(f"Preview of {source}").classes("text-lg font-bold mb-2")
            ui.markdown(f"**Document ID(s):** {', '.join(doc_ids)}").classes(
                "p-2 rounded border shadow mb-2"
            )

            for c in [repr(h) for h in chunks[:5]]:  # show only first few
                ui.markdown(c).classes("p-2 border rounded shadow mb-2")
        ui.notify("Preview ready. Confirm to store.")

    ui.upload(
        on_upload=partial(handle_upload, callback=load_preview),
        auto_upload=True,
        label="Upload PDF or DOCX",
    )

    def confirm_upload():
        num_chunks, doc_ids = commit_chunks(
            preview_state["chunks"], preview_state["label"]
        )
        doc_ids_str = ", ".join(doc_ids)
        ui.notify(f"Stored {num_chunks} chunks (doc_id: {doc_ids_str})")

    ui.button("Confirm Upload", on_click=confirm_upload)

    ui.separator()

    # Deletion
    delete_input = ui.input("Delete by doc_id").classes("w-full")

    def delete_by_doc_id():
        collection.delete(where={"doc_id": delete_input.value})
        ui.notify(f"Deleted documents with doc_id: {delete_input.value}")

    ui.button("Delete Document", on_click=delete_by_doc_id)

    # Clear all data
    ui.button("Clear All Data", on_click=clear_all_data)

    ui.separator()
    f_source = ui.input("Filter by Source").classes("w-full")
    f_doc = ui.input("Filter by doc_id").classes("w-full")
    f_date = ui.input("Upload Date (ISO)").classes("w-full")

    def run_search():
        DISTANCE_THRESHOLD = 0.7  # Define your similarity threshold

        if not q.value.strip():
            ui.notify("Please enter a search query.")
            return
        where = {}
        if f_source.value:
            where["source"] = f_source.value
        if f_doc.value:
            where["doc_id"] = f_doc.value
        if f_date.value:
            where["uploaded_at"] = {"$gte": f_date.value}

        res = collection.query(
            query_texts=q.value,
            n_results=20,
            where=where if where else None,
        )

        results.clear()
        with results:
            if res["documents"]:
                for doc, distance in zip(res["documents"][0], res["distances"][0]):
                    if distance <= DISTANCE_THRESHOLD:
                        # Display the document only if it meets the similarity threshold
                        ui.markdown(f"{doc} ({distance})").classes(
                            "p-2 border rounded mb-2"
                        )
                if not any(
                    distance <= DISTANCE_THRESHOLD for distance in res["distances"][0]
                ):
                    ui.markdown(
                        "No results found below the distance threshold."
                    ).classes("p-2")
            else:
                ui.markdown("No results found.").classes("p-2")

    q = ui.input("Search Query").classes("w-full")
    ui.button("Search", on_click=run_search)
    results = ui.column().classes("mt-4")

    # List all documents
    ui.separator()
    ui.label("List all documents").classes("text-lg font-bold mb-2")
    all_docs_button = ui.button(
        "Show All Documents", on_click=lambda: load_all_docs()
    ).classes("mb-2")

    all_docs = ui.column().classes("mt-4")

    def load_all_docs():
        all_docs.clear()
        docs = collection.get(include=["metadatas"])
        # output the document id, date, label, and source
        if docs["ids"]:
            for uid, meta in zip(docs["ids"], docs["metadatas"]):
                ui.markdown(
                    f"**UID:** {uid}\n"
                    f"**Document ID:** {meta['doc_id']}\n"
                    f"**Source:** {meta['source']}\n"
                    f"**Uploaded At:** {meta['uploaded_at']}\n"
                    f"**Label:** {meta['source']}"
                ).classes("p-2 border rounded mb-2")
        else:
            ui.markdown("No documents found in the database.").classes("p-2")


if __name__ in {"__main__", "__mp_main__"}:
    from rich.traceback import install

    install()
    ui.run(dark=True)
