import logging
import sys
from pathlib import Path
from typing import Annotated, Literal

# Patch BEFORE any import that touches Cognee logging
import noop_logging_utils
from config import settings
from mchat.model_manager import ModelManager
from mchat.tool_utils import BaseTool

# Hack - disable cognee logging, needs to be done before import
sys.modules["cognee.shared.logging_utils"] = noop_logging_utils

import cognee
from cognee.api.v1.search import SearchType

logger = logging.getLogger(__name__)

# Hack - Fix bad handler behavior from Cognee
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.terminator = "\n"


class CogneeSearchTool(BaseTool):
    name = "cognee_search"
    description = (
        "Searches a document database for information based on a query. "
        "Use this tool to find relevant data or insights from the system."
    )

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
                "understandign relationships between concepts, exploring the structure of the knowledge graph.\n. "
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

    def __init__(self, corpus: str = "default") -> None:
        super().__init__()

        # Check to see if corpus is a valid absolute path, if not, check
        # the copora path in the settings
        corpus_path = Path(corpus)
        if not corpus_path.is_absolute() or not corpus_path.exists():
            base_path = Path(settings.defaults.rag_corpora_path)
            corpus_path = (base_path / corpus).resolve()

        if not corpus_path.exists():
            raise ValueError(f"Corpus path {corpus_path} does not exist.")

        mm = ModelManager()
        rag_model = settings.defaults.rag_llm_model
        api_key = mm.config[rag_model].api_key
        cognee.config.set_llm_api_key(api_key)
        cognee.config.system_root_directory(corpus_path)
