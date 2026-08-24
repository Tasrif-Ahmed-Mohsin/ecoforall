"""
Root Application Launcher for Quad-Domain Chat Studio
------------------------------------------------------
Executes the upgraded projectresearch/scripts/web_app.py 4D Chat-First Intelligence Console.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_RESEARCH = ROOT / "projectresearch"
sys.path.insert(0, str(PROJECT_RESEARCH))
sys.path.insert(0, str(PROJECT_RESEARCH / "scripts"))

# Import and execute the 4D Chat Console main function
from projectresearch.scripts.web_app import main

if __name__ == "__main__":
    main()
