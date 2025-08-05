# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= poetry run python -m sphinx
SOURCEDIR    = docs
BUILDDIR     = docs/_build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Build the HTML documentation
html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Documentation built! Open docs/_build/html/index.html to view."

# Clean the build directory
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILDDIR)/*

# Serve the documentation locally
serve: html
	@echo "Starting documentation server..."
	@poetry run python serve_docs.py

# Install documentation dependencies
install-docs:
	@echo "Installing documentation dependencies via Poetry..."
	@poetry add --group dev sphinx sphinx-rtd-theme

# Quick rebuild and serve
dev: clean html serve

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
