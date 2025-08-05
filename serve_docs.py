#!/usr/bin/env python3
"""Simple script to serve the Sphinx documentation locally."""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def serve_docs(port=8000):
    """Serve the documentation on localhost."""
    docs_dir = Path(__file__).parent / "docs" / "_build" / "html"
    
    if not docs_dir.exists():
        print("❌ Documentation not found. Please build it first with:")
        print("cd docs && python -m sphinx -b html . _build/html")
        sys.exit(1)
    
    os.chdir(docs_dir)
    
    handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"✅ Serving documentation at {url}")
        print("Press Ctrl+C to stop the server")
        
        # Try to open browser automatically
        try:
            webbrowser.open(url)
        except Exception:
            pass
        
        httpd.serve_forever()

if __name__ == "__main__":
    serve_docs()
