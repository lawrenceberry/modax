"""Sphinx configuration for the modax documentation.

Pages are MyST Markdown; the API reference is autodoc. autodoc imports the
package, so the docs environment installs modax with its CPU dependencies:
every module imports without a GPU, since nothing touches the driver until a
kernel launches.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "modax"
author = "Lawrence Berry"
copyright = "Lawrence Berry"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

exclude_patterns = ["_build"]
# layout.html there removes the theme's top navbar.
templates_path = ["_templates"]

# -- MyST ---------------------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",  # ::: fences, for sphinx-design's grids and dropdowns
    "deflist",
    "dollarmath",  # $...$ and $$...$$, which the pages and README use
    "html_image",
]
# `[text](page.md#a-heading)` links resolve to headings up to this depth.
myst_heading_anchors = 3

# -- autodoc ------------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
# A dataclass's docstring and its __init__'s render as one.
autoclass_content = "both"
autodoc_typehints = "signature"
# Docstrings are Google style, with a few NumPy-style sections.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Single backticks in a docstring are code, linked when the name resolves.
default_role = "py:obj"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

# -- HTML ---------------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "modax"
html_logo = "assets/modax.svg"
html_static_path: list[str] = []
html_theme_options = {
    "repository_url": "https://github.com/lawrenceberry/modax",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "show_toc_level": 2,
}
