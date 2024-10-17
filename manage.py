#!/usr/bin/env python
import os
import sys

def main():
    os.environ.setdefault('FLASK_APP', 'app.py')  # Adjust if you're using Flask
    try:
        from flask.cli import main as flask_main  # For Flask projects
    except ImportError:
        raise ImportError("Flask is not installed.")
    flask_main()

if __name__ == '__main__':
    main()
