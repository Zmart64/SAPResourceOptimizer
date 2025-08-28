import sys
from pathlib import Path
try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover - fallback for Python <3.11
    try:
        import tomli as tomllib  # type: ignore
    except Exception:
        tomllib = None  # type: ignore

# Project metadata
project = 'Resource Prediction'
author = 'Project Contributors'

# Derive release/version from pyproject.toml if available
try:
    pyproject_path = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    if tomllib is not None and pyproject_path.exists():
        with pyproject_path.open('rb') as f:
            data = tomllib.load(f)
        release = data.get('tool', {}).get('poetry', {}).get('version', '0.1.0')
    else:
        release = '0.1.0'
except Exception:
    release = '0.1.0'

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# Mock heavy imports for reliable docs builds without full ML stack
# Avoid mocking numpy to prevent type-hint evaluation errors like
# "unsupported operand type(s) for |: 'ndarray' and 'NoneType'" when modules
# use PEP 604 unions (e.g., np.ndarray | None). Other heavy libs remain mocked
# to keep docs builds lightweight.
autodoc_mock_imports = [
    'pandas', 'sklearn', 'matplotlib', 'seaborn', 'optuna',
    'xgboost', 'lightgbm', 'catboost', 'tqdm', 'joblib', 'streamlit', 'altair'
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Use a better theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
