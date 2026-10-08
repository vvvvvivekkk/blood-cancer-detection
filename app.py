"""`python app.py` == `python run.py`: starts the web app at http://localhost:8000 (same flags)."""
import sys

from run import main

if __name__ == "__main__":
    sys.exit(main())
