# CODEHILITE_CSS_URL = f'/_nicegui/{__version__}/codehilite.css'
import asyncio
import json
import re
import time
from functools import lru_cache
from logging import getLogger
from typing import (
    Any,
    Self,
    cast,
)

from bs4 import BeautifulSoup
from fastapi.responses import PlainTextResponse

# import markdown2
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from mdit_py_plugins.deflist import deflist_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin
from mdit_py_plugins.texmath import texmath_plugin
from nicegui import __version__, core, ui

# from .mermaid import Mermaid
from nicegui.element import Element
from nicegui.elements.mixins.content_element import ContentElement
from nicegui.timer import Timer as timer
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

logger = getLogger(__name__)

CODEHILITE_CSS_URL = f"/_nicegui/{__version__}/codehilite.css"
SMART_CODE_JS_PATH = "/_nicegui/components/_code.js"


class MyMarkdown(MarkdownIt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def highlight_code(code, name, attrs):
    """Highlight a block of code"""

    if attrs:
        logger.debug(f"Ignoring {attrs=}")

    try:
        lexer = get_lexer_by_name(name)
    except Exception as e:
        logger.debug(f"Failed to get lexer for {name}: {e}, defaulting 'text'")
        lexer = get_lexer_by_name("text")
    formatter = HtmlFormatter(cssclass="codehilite")
    out = highlight(code, lexer, formatter)
    return out

    # return highlight(code, lexer, formatter)


@lru_cache
def get_markdown_parser() -> MarkdownIt:
    md = MyMarkdown(
        config="js-default",
        options_update={
            "html": True,
            "linkify": True,
            "typographer": True,
            "breaks": True,
            "highlight": highlight_code,
        },
    )
    return (
        md.use(tasklists_plugin)
        .use(deflist_plugin)
        .use(footnote_plugin)
        .use(texmath_plugin)
    )


def register_codehilite_css(
    light_style: str = "monokai", dark_style: str = "github-dark"
) -> None:
    """(Re)register the codehilite CSS route with custom styles."""
    # Remove existing route if any
    existing_routes = [
        r for r in core.app.routes if getattr(r, "path", None) == CODEHILITE_CSS_URL
    ]
    for route in existing_routes:
        core.app.routes.remove(route)

    # Generate new CSS
    light_css = HtmlFormatter(nobackground=True, style=light_style).get_style_defs(
        ".codehilite"
    )
    dark_css = HtmlFormatter(nobackground=True, style=dark_style).get_style_defs(
        ".body--dark .codehilite"
    )

    # Register the new route
    core.app.get(CODEHILITE_CSS_URL)(
        lambda: PlainTextResponse(light_css + dark_css, media_type="text/css")
    )


def html_table_to_table_dict(html: str) -> dict[str, list[dict]]:
    """Identify 'columns' and 'rows' for ui.table()."""
    soup = BeautifulSoup(html, "html.parser")

    # Extract headers and normalize to field names
    headers = [th.get_text(strip=True) for th in soup.find_all("th")]
    field_names = [re.sub(r"\W+", "_", h.strip().lower()) for h in headers]

    # Extract raw rows of cell data
    raw_rows = [
        [td.get_text(strip=True) for td in tr.find_all("td")]
        for tr in soup.find_all("tr")[1:]  # Skip header row
    ]

    # Transpose cell data to get column-wise view
    column_data = list(zip(*raw_rows, strict=False))  # avoid IndexError for short rows

    def is_numeric_column(values: list[str]) -> bool:
        return all(re.match(r"^-?\d+(\.\d+)?$", v) for v in values if v.strip())

    # Build columns metadata
    columns = []
    for i, (field, label) in enumerate(zip(field_names, headers, strict=True)):
        values = column_data[i] if i < len(column_data) else []
        numeric = is_numeric_column(values)
        column = {
            "name": field,
            "label": label,
            "field": field,
            "align": "right" if numeric else "left",
            "sortable": len(set(values)) > 1,
        }
        columns.append(column)

    # Build row dicts
    rows = []
    for cells in raw_rows:
        row = {
            field: cell
            for field, cell in zip(field_names, cells, strict=True)
            if cell  # skip empty values
        }
        rows.append(row)

    return {"columns": columns, "rows": rows}


def print_tree_with_lineage(node, lineage="0"):
    """Print the markdown tree structure with hierarchical lineage identifiers."""
    logger.debug(f"{lineage}: {node} {node.content if lineage != '0' else '--'}")

    for i, child in enumerate(node.children):
        child_lineage = f"{lineage}.{i}"
        print_tree_with_lineage(child, child_lineage)


class SmartMarkdown(
    Element, component="markdown.js", default_classes="nicegui-markdown"
):  # todo - consider not using markdown.js and usign a local version
    def __init__(
        self,
        content: str = "",
        *,
        # extras: list[str] = ["fenced-code-blocks", "tables"],  # noqa: B006
        extras: list[str] = None,
        **kwargs,
    ) -> None:
        """Markdown Element
        Renders Markdown onto the page.
        """
        if extras is None:
            extras = []
        self.content = content
        self.extras = extras[:]
        super().__init__(**kwargs)
        self._props["codehilite_css_url"] = CODEHILITE_CSS_URL
        register_codehilite_css()
        self._handle_content_change(self.content)

    def _handle_content_change(self, content: str) -> None:
        # parse the content into tokens and ast
        md = get_markdown_parser()
        tokens = md.parse(content)
        nodes = SyntaxTreeNode(tokens)
        # print_tree_with_lineage(nodes)

        # parse through the AST nodes and create a map on what to render
        map = []
        for node in nodes.children:
            if node.type == "fence":
                if node.info.startswith("mermaid"):
                    map.append((node, "mermaid"))
                    continue
                map.append((node, "fence"))
            elif node.type == "table":
                map.append((node, "table"))
            else:
                map.append(
                    (md.renderer.render(node.to_tokens(), md.options, {}), "html")
                )
        # render the map
        for element, element_type in map:
            if element_type == "html":
                ui.html(element).classes("list-decimal")
            elif element_type == "fence":
                SmartCode(element.content, language=element.info)
            elif element_type == "mermaid":
                ui.mermaid(element.content)
            elif element_type == "table":
                dimensions = html_table_to_table_dict(
                    md.renderer.render(element.to_tokens(), md.options, {})
                )
                ui.table(**dimensions)


class SmartCode(
    ContentElement, component="smart_markdown.js", default_classes="nicegui-code"
):  # todo fsm
    def __init__(
        self, content: str = "", *, language: str | None = "python", **kwargs: Any
    ) -> None:
        """SmartCode

        This element displays a code block with syntax highlighting.

        In secure environments (HTTPS or localhost), a copy button is displayed to copy the code to the clipboard.

        :param content: code to display
        :param language: language of the code (default: "python")
        """
        # These need to be set beofore super().__init__ as __init__ will call
        # _handle_content_change which relies on these properties to be set.
        self.content = content
        self.language = language
        self.code = None  # Placeholder for the code element
        super().__init__(content=content, **kwargs)

        try:
            self.lexer = get_lexer_by_name(self.language)
        except Exception as e:
            logger.debug(
                f"Failed to get lexer for {self.language}: {e}, defaulting 'text'"
            )
            self.lexer = get_lexer_by_name("text")

        self._props["codehilite_css_url"] = CODEHILITE_CSS_URL
        register_codehilite_css()

        # Set up the UI components
        with self:
            self.code = ui.html().classes("overflow-auto")
            self.copy_button = (
                ui.button(icon="content_copy", on_click=self.show_checkmark)
                .props("round flat size=sm")
                .classes("absolute right-2 top-2 opacity-20 hover:opacity-80")
                .on(
                    "click",
                    js_handler=(
                        f"() => navigator.clipboard.writeText("
                        f"{json.dumps(self.content)})"
                    ),
                )
            )

        self._handle_content_change(self.content)

        self._last_scroll: float = 0.0
        self.code.on("scroll", self._handle_scroll)
        timer(0.5, self._update_copy_button)

        self.client.on_connect(
            lambda: self.client.run_javascript(f"""
            if (!navigator.clipboard) getHtmlElement({self.copy_button.id}).style.display = 'none';
        """)
        )

    async def show_checkmark(self) -> None:
        """Show a checkmark icon for 3 seconds."""
        self.copy_button.props("icon=check")
        await asyncio.sleep(3.0)
        self.copy_button.props("icon=content_copy")

    def _handle_scroll(self) -> None:
        self._last_scroll = time.time()

    def _update_copy_button(self) -> None:
        self.copy_button.set_visibility(time.time() > self._last_scroll + 1.0)

    def _handle_content_change(self, content: str) -> None:
        """Handle content change and update the syntax highlighting."""
        # this method is run early by the parent before the UI compoenents
        # are created, we will call it in __init__ when we're ready
        if not self.code:
            return

        content = remove_indentation(content)

        html = highlight(
            content,
            self.lexer,
            HtmlFormatter(
                cssclass="codehilite", linenos="table"
            ),  # Add linenos if needed,
        )
        self.content = content
        self.code.set_content(html)
        self.update()


def remove_indentation(text: str) -> str:
    """
    Remove indentation from a multi-line string based on the indentation
    of the first non-empty line.
    """
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""
    indentation = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indentation:] for line in lines)
