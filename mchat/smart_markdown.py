import asyncio
import json
import re
import time
from functools import lru_cache
from logging import getLogger
from typing import (
    Any,
)

from bs4 import BeautifulSoup
from fastapi.responses import PlainTextResponse

# import markdown2
from markdown_it import MarkdownIt
from markdown_it.renderer import RendererHTML
from markdown_it.token import Token
from markdown_it.tree import SyntaxTreeNode
from mdit_py_plugins.amsmath import amsmath_plugin
from mdit_py_plugins.deflist import deflist_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin

# from mdit_py_plugins.texmath import tex2svg
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

    """Markdown Element that renders advanced Markdown with code highlighting, mermaid, tables, etc."""

    def __init__(
        self, content: str = "", *, extras: list[str] = None, **kwargs
    ) -> None:
        if extras is None:
            extras = []
        self.content = ""
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
        # self._handle_content_change(self.content)

    def _retired_handle_content_change(self, content: str) -> None:
        """Parse and render the Markdown content into NiceGUI elements."""
        tokens = self.md.parse(content)
        node_tree = SyntaxTreeNode(tokens)

        # Clear any existing elements before re-rendering
        self.clear()

        # Collect elements to render in a list
        render_map = []
        for node in node_tree.children:
            if node.type == "fence":
                # Check for mermaid code blocks
                if node.info.startswith("mermaid"):
                    render_map.append((node, "mermaid"))
                else:
                    render_map.append((node, "code_block"))
            elif node.type == "table":
                render_map.append((node, "table"))
            else:
                # Default: Render as HTML using markdown-it's renderer
                html_content = self.md.renderer.render(
                    node.to_tokens(), self.md.options, {}
                )
                render_map.append((html_content, "html"))

        # Render the collected items into the NiceGUI UI
        with self:
            for element, element_type in render_map:
                if element_type == "html":
                    ui.html(element).classes("list-decimal")
                elif element_type == "code_block":
                    # Add your custom code element or NiceGUI code element
                    ui.code(element.content, language=element.info)
                elif element_type == "mermaid":
                    ui.mermaid(element.content)
                elif element_type == "table":
                    table_html = self.md.renderer.render(
                        element.to_tokens(), self.md.options, {}
                    )
                    dimensions = html_table_to_table_dict(table_html)
                    with ui.element("div").classes("w-min"):
                        ui.table(**dimensions).classes("m-2 table-auto").props("")
                else:
                    logger.debug(f"Unknown element type: {element_type}")

    def replace(self, content: str) -> None:
        """Replace the current content and re-render."""
        self.content = content
        self._handle_content_change(self.content)

    @trace(logger)
    def append(self, new_content: str) -> None:
        self.content += new_content
        new_tokens = self.md.parse(self.content)
        new_ast = SyntaxTreeNode(new_tokens)

        # new_top_nodes = new_ast.children
        new_top_nodes = [n for n in new_ast.children if n.type != "footnote_block"]

        prev_count = len(self.top_level_nodes)
        new_count = len(new_top_nodes)

        # Step 1: Re-render the last known top-level node (if it exists)
        if prev_count > 0:
            path_str = self._stringify_path([0, prev_count - 1])
            last_node = new_top_nodes[prev_count - 1]  # Use the new node
            # Testing
            # last_node = self.top_level_nodes[-1]
            self._render_node(last_node, path_str)

        # Step 2: Render any new top-level nodes
        for i in range(prev_count, new_count):
            node = new_top_nodes[i]
            path_str = self._stringify_path([0, i])
            self._render_node(node, path_str)

        # Step 3: Update tracker
        self.top_level_nodes = new_top_nodes

        # Step 4: Handle footnote_block if it exists
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

        # Check if there is an element for the node - if so, update it
        element = self.node_elements.get(path_str, None)
        if element:
            if isinstance(element, ui.html):
                html = self.md.renderer.render(node.to_tokens(), self.md.options, {})
                element.set_content(html)
                if _needs_mathjax(html):
                    self._typeset_math(element.id)
            elif isinstance(element, SmartCode):
                element.set_content(node.content, language=node.info)
            else:
                raise ValueError(f"Unknown element type for node: {type(element)}")
            return

        if node.type == "fence":
            with self:
                element = SmartCode(node.content, language=node.info).classes("my-2")
        else:
            with self:
                html = self.md.renderer.render(node.to_tokens(), self.md.options, {})
                element = ui.html(html).classes("w-full my-2")
                if _needs_mathjax(html):
                    self._typeset_math(element.id)
        self.node_elements[path_str] = element

    def _stringify_path(self, path: list[int]) -> str:
        return ".".join(str(p) for p in path)

    def _typeset_math(self, element_id: int) -> None:
        def trigger_mathjax():
            local_element_id = "c" + str(element_id)  # Get the component's element ID
            escaped_element_id = json.dumps(local_element_id)

            # JavaScript code to re-typeset math in the specific element
            js_code = f"""
            (function retryTypeset(attempts) {{
                if (typeof attempts === 'undefined') {{
                    attempts = 0;
                }}

                // Check if MathJax is loaded
                if (typeof window.MathJax === 'undefined') {{
                    console.error("MathJax is not loaded. Aborting further attempts.");
                    return; // Exit if MathJax is not loaded
                }} else {{
                    //console.log("MathJax is loaded:", window.MathJax);
                }}

                var element = document.getElementById({escaped_element_id});
                if (window.MathJax.typesetPromise && element && attempts < 50) {{
                    //console.log("MathJax ready. Typesetting math in element:", element);
                    MathJax.typesetPromise([element]).then(() => {{
                        //console.log("Math typeset successfully in element.");
                    }}).catch((error) => {{
                        console.error("Error in typesetting:", error);
                    }});
                }} else if (attempts < 50) {{
                    //console.log(`MathJax not ready or element ${escaped_element_id} not found, retrying...`);
                    setTimeout(() => retryTypeset(attempts + 1), 100);
                }} else {{
                    console.error("MathJax typeset retry attempts exceeded.");
                }}
            }})(0);
            """
            ui.run_javascript(js_code)  # Execute the JavaScript code

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
            ui.add_head_html(smarthighlight_style)
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
            lambda: self.client.run_javascript(f"""
            if (!navigator.clipboard) getHtmlElement({self.copy_button.id}).style.display = 'none';
        """)
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
        except Exception as e:
            # logger.debug(
            #     f"Failed to get lexer for {self.language}: {e}, defaulting 'text'"
            # )
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
