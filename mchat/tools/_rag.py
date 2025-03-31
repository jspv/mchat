import logging
from typing import Annotated

import chromadb

from mchat.tool_utils import BaseTool

logger = logging.getLogger(__name__)
logger.warning("Starting the RAG tool")

client = chromadb.PersistentClient(
    path="./chroma_persist",
)


class Rag(BaseTool):
    name = "ragtest"
    description = "Get information about Pokémon card game rules"

    def run(
        self,
        query: Annotated[str, "Search query"],
        num_results: Annotated[
            int, "Number of results to fetch", "Default(value=5)"
        ] = 5,
    ) -> Annotated[list[dict], "A list of results from the search query"]:
        """
        Search the pokemon card game rules using the provided query.

        Returns:
            list[dict] : A list of excerpts from various Pokémon card game rules
        """
        logger.debug("Running RAG tool with query: %s", query)
        collection = client.get_collection(name="docs")

        results = collection.query(
            query_texts=query,
            n_results=num_results,
            include=["documents", "metadatas"],
        )

        logger.debug(
            "Query results: %s",
            results,
        )

        # Extract the documents from the results
        documents = results["documents"][0]

        # Format the documents into a list of dictionaries
        formatted_results = [
            {
                "content": doc,
                "source": f"Document {results['metadatas'][0][i]['filename']}",
            }
            for i, doc in enumerate(documents)
        ]

        return formatted_results
