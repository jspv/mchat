import logging
import sys
from pathlib import Path
from typing import Annotated, Literal

# Hack - Patch BEFORE any import that touches Cognee logging
import noop_logging_utils

sys.modules["cognee.shared.logging_utils"] = noop_logging_utils

import cognee
from cognee.api.v1.search import SearchType

from config import settings
from mchat.model_manager import ModelManager
from mchat.tool_utils import BaseTool

logger = logging.getLogger(__name__)

# Hack - Fix bad handler behavior from Cognee
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.terminator = "\n"

# If using a factory, list the name, description, and corpus for the tool.
tools = [
    (
        "name",
        "description",
        "corpus",
    )
]


class CogneeSearchTool(BaseTool):
    async def run(
        self,
        query_text: Annotated[
            str, "The query to use for semantic and graph search"
        ] = "",
        query_type: Annotated[
            Literal["SUMMARIES", "INSIGHTS", "CHUNKS"],
            (
                "Type of search to perform.  SUMMMARIES: retrieves summarized information. "
                "Best for: getting concise overviews of topics, summarizing a large amount of information, "
                "quick understanding of complex subjects.\nINSIGHTS: discovers relationships and connections "
                "between entities in the knowlege graph.  Best for: discovering how entities are connected, "
                "understanding relationships between concepts, exploring the structure of the knowledge graph.\n. "
                "Best for: extracting key insights, understanding trends, and making data-driven decisions.\nCHUNKS: "
                "retrieves specific chunks of data.  Best for: detailed analysis, finding specific information, "
                "and in-depth research."
            ),
        ] = "CHUNKS",
        top_k: Annotated[int, "Number of top results to return"] = 10,
    ) -> list:
        """
        Perform a search in the Cognee database.

        Args:
            query_text (str): The query to use for the search.
            query_type (str): Type of search to perform (SUMMARIES, INSIGHTS, CHUNKS).
            top_k (int): Number of top results to return.
        """
        import litellm

        litellm._turn_on_debug()

        logger.debug(
            f"received args: query_text={query_text}, query_type={query_type}, top_k={top_k}"
        )

        query_type_map = {
            "SUMMARIES": SearchType.SUMMARIES,
            "INSIGHTS": SearchType.INSIGHTS,
            "CHUNKS": SearchType.CHUNKS,
        }

        query_type = query_type_map.get(query_type)

        # Placeholder for actual search logic
        results = await cognee.search(
            query_text=query_text,
            query_type=query_type,
            # top_k=top_k,
        )
        logger.debug(f"Search results: {results}")
        return results

    def __init__(self, name: str, description: str, corpus: str) -> None:
        super().__init__()
        self.name = name
        self.description = description

        # Check to see if corpus is a valid absolute path, if not, check
        # the copora path in the settings
        corpus_path = Path(corpus)
        if not corpus_path.is_absolute():
            corpus_path = Path(settings.defaults.rag_corpora_path) / corpus
        corpus_path = corpus_path.resolve()
        if not corpus_path.exists():
            raise ValueError(f"Corpus path {corpus_path} does not exist.")

        if not corpus_path.exists():
            raise ValueError(f"Corpus path {corpus_path} does not exist.")

        mm = ModelManager()
        rag_model = settings.defaults.rag_llm_model
        api_key = mm.config[rag_model].api_key
        cognee.config.set_llm_api_key(api_key)
        cognee.config.system_root_directory(corpus_path)


class CogneeSearchToolFactory:
    """
    Factory class to create instances of CogneeSearchTool with a specific corpus.
    """

    @staticmethod
    def create_tool(name: str, description: str, corpus: str) -> CogneeSearchTool:
        """
        Create an instance of CogneeSearchTool with the specified corpus.

        Args:
            corpus (str): The corpus to use for the search tool.

        Returns:
            CogneeSearchTool: An instance of the search tool.
        """
        return CogneeSearchTool(name, description, corpus)


USE_FACTORY = True
FACTORY = CogneeSearchToolFactory
