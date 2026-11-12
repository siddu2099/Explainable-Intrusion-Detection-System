"""
Backward-compatibility wrapper for projectv5.py
Forwards execution to main.py
"""

from main import run_pipeline

if __name__ == "__main__":
    run_pipeline()
