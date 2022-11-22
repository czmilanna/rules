from cacp.util import seed_everything

from research import RESULT_DIR, DATA_DIR
from research.comparison import process_comparison
from research.comparison_of_rules import process_comparison_of_rules
from research.setup import classifiers

if __name__ == '__main__':
    seed_everything(1)
    process_comparison(classifiers, DATA_DIR, RESULT_DIR)
    process_comparison_of_rules(classifiers, DATA_DIR, RESULT_DIR.joinpath('rules'))
