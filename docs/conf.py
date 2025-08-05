import os
import sys
from pathlib import Path

project = 'Resource Prediction'
author = 'Your Name'
release = '0.1.0'

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []
html_theme = 'alabaster'
