import importlib.metadata, os, sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

project = "qsimx"
copyright = f"{datetime.now().year}, qsimx"
author = "qsimx contributors"

release = importlib.metadata.version("qsimx") if importlib.util.find_spec("qsimx") else "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

autodoc_member_order = "bysource"

html_theme = "sphinx_rtd_theme"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"] 