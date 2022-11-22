from pathlib import Path

RESEARCH_DIR = Path(__file__).parent
DATA_DIR = RESEARCH_DIR.parent.joinpath('data')
RESULT_DIR = RESEARCH_DIR.joinpath('result')
