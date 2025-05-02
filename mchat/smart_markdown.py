import asyncio
import json
import re
import textwrap
import time
from functools import lru_cache
from logging import getLogger
from typing import (
    Any,
)

from bs4 import BeautifulSoup
from fastapi.responses import PlainTextResponse
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from mdit_py_plugins.deflist import deflist_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin
from nicegui import __version__, core, ui

# from .mermaid import Mermaid
from nicegui.element import Element
from nicegui.elements.mixins.content_element import ContentElement
from nicegui.timer import Timer as timer
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

from .logging_config import trace

logger = getLogger(__name__)

CODEHILITE_CSS_URL = f"/_nicegui/{__version__}/codehilite.css"
SMART_CODE_JS_PATH = "/_nicegui/components/_code.js"

smarthighlight_style = """
<style>
.nicegui-smartcode {
  position: relative;
  background-color: rgba(127, 159, 191, 0.1);
  border: 1pt solid rgba(127, 159, 191, 0.15);
  box-shadow: 0 0 0.5em rgba(127, 159, 191, 0.05);
  border-radius: 0.25rem;
}
.nicegui-smartcode .codehilite {
  /*padding: 0 0.5rem;*/
}

.nicegui-smartcode td {
    padding: 0;
    border: 0
}
</style>
"""

definition_list_css = """
    <style>
    dl {
        margin: 1em 0;
    }
    dt {
        display: inline;
        font-weight: bold;
    }
    dd {
        display: inline;
        margin-left: 0.5em;
        margin-right: 1em;
    }
    dd::after {
        content: "\\A";
        white-space: pre;
    }
    </style>
    """

xmathjax_js = """
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    packages: ['base', 'ams']
  },
  options: {
    processHtmlClass: 'math',  // allow processing of <div class="math block">
    ignoreHtmlClass: '.*'
  }
};
</script>
<!-- MathJax -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"
    integrity="sha512-tiaNmAzpy3KcgtuiLwT9WSlSsqGqtDB5ylMwxoqG5ysNIyzkBw24k6UFTuXGgyXJLJ8aM/ho1h67NRKedPx++Q=="
    crossorigin="anonymous" referrerpolicy="no-referrer"></script>
"""

ui.add_body_html(xmathjax_js, shared=True)
ui.add_head_html(smarthighlight_style, shared=True)

MATHJAX_RETRY_SCRIPT = """
(function retryTypeset(attempts) {
    if (typeof attempts === 'undefined') {
        attempts = 0;
    }
    if (typeof window.MathJax === 'undefined') {
        console.error("MathJax is not loaded. Aborting further attempts.");
        return;
    }
    var element = document.getElementById(__ELEMENT_ID__);
    if (window.MathJax.typesetPromise && element && attempts < 50) {
        MathJax.typesetPromise([element]).then(() => {
        }).catch((error) => {
            console.error("Error in typesetting:", error);
        });
    } else if (attempts < 50) {
        setTimeout(() => retryTypeset(attempts + 1), 100);
    } else {
        console.error("MathJax typeset retry attempts exceeded.");
    }
})(0);
"""


class MyMarkdown(MarkdownIt):
    """Simple subclass to hold any custom MarkdownIt configuration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@lru_cache
def highlight_code(code: str, lang: str, attrs=None) -> str:
    """Highlight a block of code using Pygments."""
    if attrs:
        logger.debug(f"Ignoring extra attributes: {attrs}")

    try:
        lexer = get_lexer_by_name(lang)
    except Exception as e:
        logger.debug(f"Failed to get lexer for {lang}: {e}, defaulting to 'text'")
        lexer = get_lexer_by_name("text")

    formatter = HtmlFormatter(cssclass="codehilite")
    return highlight(code, lexer, formatter)


@lru_cache
def get_markdown_parser() -> MarkdownIt:
    # def math_renderer(content, _):
    #     # Return math content without escaping
    #     return rf"\[{content}\]"

    def math_inline(tokens, idx, _opts, _env):
        return f'<span class="math inline">${tokens[idx].content}$</span>'

    def math_block(tokens, idx, _opts, _env):
        return f'<div class="math block">$$\n{tokens[idx].content}\n$$</div>\n'

    """Create and return a cached MarkdownIt parser with desired plugins."""
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

    dollarmath_options = {
        "allow_blank_lines": True,
        "allow_labels": True,
        "allow_space": True,
        "allow_digits": True,
        "double_inline": False,
    }
    md.use(tasklists_plugin).use(deflist_plugin).use(footnote_plugin).use(
        dollarmath_plugin, **dollarmath_options
    )
    # .use(amsmath_plugin)

    md.renderer.rules["math_inline"] = math_inline
    md.renderer.rules["math_block"] = math_block
    return md


def register_codehilite_css(
    light_style: str = "solarized-light", dark_style: str = "monokai"
) -> None:
    """(Re)register the codehilite CSS route with custom styles."""
    # Remove existing route if any
    existing_routes = [
        r for r in core.app.routes if getattr(r, "path", None) == CODEHILITE_CSS_URL
    ]
    for route in existing_routes:
        core.app.routes.remove(route)

    # Generate new CSS from Pygments for light and dark themes
    light_css = HtmlFormatter(nobackground=True, style=light_style).get_style_defs(
        ".codehilite"
    )
    dark_css = HtmlFormatter(nobackground=True, style=dark_style).get_style_defs(
        ".body--dark .codehilite"
    )

    # Custom CSS for line numbers
    lineno_css = """
    td.linenos, td.linenos .normal, span.linenos, span.linenos.special {
        opacity: 0.5; /* 50% transparency */
        padding-right: 10px; /* Extra spacing for line numbers */
    }
    """

    # Combine and register
    core.app.get(CODEHILITE_CSS_URL)(
        lambda: PlainTextResponse(
            light_css + dark_css + lineno_css, media_type="text/css"
        )
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


class SmartMarkdown(
    Element, component="smart_markdown.js", default_classes="nicegui-markdown"
):
    _mathjax_initialized = False
    _css_initialized = False

    """NiceGUI Element that renders advanced Markdown with code highlighting,
    mermaid, tables, etc."""

    def __init__(
        self, content: str = "", *, extras: list[str] = None, **kwargs
    ) -> None:
        if extras is None:
            extras = []
        self.content = ""  # content will be added by append()
        self.extras = extras[:]
        self.top_level_nodes: list[SyntaxTreeNode] = []
        self.node_elements: dict[SyntaxTreeNode, Element] = {}
        self.footnotes_element: ui.element | None = None

        super().__init__(**kwargs)

        # Register highlight CSS
        self._props["codehilite_css_url"] = CODEHILITE_CSS_URL
        register_codehilite_css()

        if not SmartMarkdown._css_initialized:
            ui.add_css(definition_list_css)
            SmartMarkdown._css_initialized = True

        if not SmartMarkdown._mathjax_initialized:
            # handling this at module level otherwise the mathjax code wasn't in
            # place when SmartMarkdown was created after ui.run
            # ui.add_body_html(xmathjax_js, shared=True)
            SmartMarkdown._mathjax_initialized = True

        # Instantiate the parser
        self.md = get_markdown_parser()

        # Render initial content
        self.append(content)

    def replace(self, content: str) -> None:
        """Replace the current content and re-render."""
        self.content = ""
        # clear all children
        for path in list(self.node_elements):
            element = self.node_elements.pop(path)
            element.delete()
        self.top_level_nodes = []
        self.footnotes_element = None
        self.append(self.content)

    @trace(logger)
    def append(self, new_content: str) -> None:
        self.content += new_content
        new_tokens = self.md.parse(self.content)
        new_ast = SyntaxTreeNode(new_tokens)

        # Find first top-level node that has changed
        for i, new_node in enumerate(new_ast.children):
            if i >= len(self.top_level_nodes) or not self._nodes_equal(
                new_node, self.top_level_nodes[i]
            ):
                break

        # previously rendered nodes that are no longer in the new content (rare case)
        for path in list(self.node_elements):
            idx = int(path.split(".")[1])
            if idx > i:
                element = self.node_elements.pop(path)
                element.delete()

        # new_top_nodes = new_ast.children
        new_top_nodes = [n for n in new_ast.children if n.type != "footnote_block"]

        # Render all top-level nodes that have change or are new
        for j, node in enumerate(new_top_nodes[i:]):
            path_str = self._stringify_path([0, i + j])
            self._render_node(node, path_str)

        # Update tracker
        self.top_level_nodes = new_top_nodes

        # Handle footnote_block if it exists
        for node in new_ast.children:
            if node.type == "footnote_block":
                html = self.md.renderer.render(node.to_tokens(), self.md.options, {})
                if not self.footnotes_element:
                    with self:
                        self.footnotes_element = ui.html(html).classes("w-full my-2")
                else:
                    self.footnotes_element.move(self)
                    self.footnotes_element.set_content(html)
                break  # there should be only one

    @trace(logger)
    def _render_node(self, node: SyntaxTreeNode, path_str: str) -> None:
        """Render a finalized node using either code block handling or raw HTML."""

        def _needs_mathjax(html: str) -> bool:
            return 'class="math' in html  # lightweight check

        custom_blocks = ["fence"]
        # Check if there is an element for the node - if so, update it
        element = self.node_elements.get(path_str, None)
        if element:
            if isinstance(element, ui.html) and node.type not in custom_blocks:
                html = self.md.renderer.render(node.to_tokens(), self.md.options, {})
                element.set_content(html)
                if _needs_mathjax(html):
                    self._typeset_math(element.id)
                # self.node_elements[path_str] = element
                return
            elif isinstance(element, SmartCode) and node.type == "fence":
                # if type is mermaid, replace with a mermaid element
                if node.info == "mermaid":
                    element.delete()
                    with self:
                        element = ui.mermaid(node.content).classes("my-2")
                        self.node_elements[path_str] = element
                else:
                    element.set_content(node.content, language=node.info)
                return
            else:
                # block type changed, delete it and create a new one
                logger.debug(
                    f"Deleting element {element.id} for node {node.type} "
                    f"which used to be {type(element)}"
                )
                logger.debug(f"content was {element.content}, is now {node.content}")
                element.delete()

        if node.type == "fence" and node.info != "mermaid":
            with self:
                element = SmartCode(node.content, language=node.info).classes("my-2")
        elif node.type == "fence" and node.info == "mermaid":
            with self:
                element = ui.mermaid(node.content).classes("my-2")
        else:
            with self:
                html = self.md.renderer.render(node.to_tokens(), self.md.options, {})
                element = ui.html(html).classes("w-full my-2")
                if _needs_mathjax(html):
                    self._typeset_math(element.id)

        self.node_elements[path_str] = element

    def _nodes_equal(self, node1: SyntaxTreeNode, node2: SyntaxTreeNode) -> bool:
        """Check if two SyntaxTreeNodes are equal"""
        if node1.type != node2.type:
            return False
        if node1.content != node2.content:
            return False
        if len(node1.children) != len(node2.children):
            return False
        return all(
            self._nodes_equal(c1, c2)
            for c1, c2 in zip(node1.children, node2.children, strict=True)
        )

    def _stringify_path(self, path: list[int]) -> str:
        return ".".join(str(p) for p in path)

    def _typeset_math(self, element_id: int) -> None:
        def trigger_mathjax():
            local_element_id = "c" + str(element_id)  # Get the component's element ID
            escaped_element_id = json.dumps(local_element_id)

            # JavaScript code to re-typeset math in the specific element
            js_code = MATHJAX_RETRY_SCRIPT.replace("__ELEMENT_ID__", escaped_element_id)
            ui.run_javascript(js_code)

        with self:
            # Schedule the JavaScript call after the next render frame
            ui.timer(0.1, trigger_mathjax, once=True)


class SmartCode(
    ContentElement, component="smart_markdown.js", default_classes="nicegui-smartcode"
):  # todo fsm
    """SmartCode Element that displays a code block with syntax highlighting."""

    _css_initialized = False
    _codehilight_registered = False

    def __init__(
        self, content: str = "", *, language: str | None = "python", **kwargs: Any
    ) -> None:
        """SmartCode

        This element displays a code block with syntax highlighting.

        In secure environments (HTTPS or localhost), a copy button is displayed
        to copy the code to the clipboard.

        :param content: code to display
        :param language: language of the code (default: "python")
        """
        # These need to be set before super().__init__ as __init__ will call
        # _handle_content_change which relies on these properties to be set.
        self.content = content
        self.language = language
        self.lexer = None
        self.code = None  # Placeholder for the code element
        if not SmartCode._css_initialized:
            # ui.add_css(smarthighlight_style)
            # ui.add_head_html(smarthighlight_style)
            SmartCode._css_initialized = True
        super().__init__(content=content, **kwargs)

        self._set_lexer(self.language)

        self._props["codehilite_css_url"] = CODEHILITE_CSS_URL
        if not SmartCode._codehilight_registered:
            register_codehilite_css()
            SmartCode._codehilight_registered = True

        # Set up the UI components
        with self:
            self.code = ui.html().classes("overflow-auto")
            self.copy_button = (
                ui.button(icon="content_copy", on_click=self.show_checkmark)
                .props("round flat size=sm")
                .classes("absolute right-2 top-2 opacity-20 hover:opacity-80")
            )

        self._handle_content_change(self.content)

        self._last_scroll: float = 0.0
        self.code.on("scroll", self._handle_scroll)
        # timer(0.5, self._update_copy_button)

        self.client.on_connect(
            lambda: self.client.run_javascript(
                textwrap.dedent(f"""
                    if (!navigator.clipboard) {{
                        getHtmlElement({self.copy_button.id}).style.display = 'none';
                    }}
                """)
            )
        )

    async def show_checkmark(self) -> None:
        """Show a checkmark icon for 3 seconds."""
        self.copy_button.props("icon=check")
        await asyncio.sleep(3.0)
        self.copy_button.props("icon=content_copy")

    def _set_lexer(self, lang: str) -> Any:
        """Get the lexer for the specified language."""
        try:
            lexer = get_lexer_by_name(lang)
        except Exception:
            lexer = get_lexer_by_name("text")
        self.language = lang
        self.lexer = lexer
        return self.lexer

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
        self._update_copy_button_handler()
        self.update()

    def _update_copy_button_handler(self):
        self.copy_button.on(
            "click",
            js_handler=(
                f"() => navigator.clipboard.writeText({json.dumps(self.content)})"
            ),
        )

    def set_content(self, content: str, language: str = None) -> None:
        """Set the content of the code block and re-render."""
        if language and language != self.language:
            self._set_lexer(language)
        super().set_content(content)


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
