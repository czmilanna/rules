import pandas as pd
from cacp.plot import process_comparison_results_plots
from cacp.result import process_comparison_results
from cacp.time import process_times
from cacp.util import seed_everything
from cacp.wilcoxon import process_wilcoxon
from cacp.winner import process_comparison_result_winners

from research import RESULT_DIR
from research.parse_rules import process_parsing_of_rules
from research.setup import classifiers

if __name__ == '__main__':
    seed_everything(1)

    result_file = RESULT_DIR.joinpath('comparison.csv')
    df = pd.read_csv(result_file)
    df['Mixed'] = 0.3 * df['AUC'] + 0.5 * df['Recall'] + 0.05 * (
            df['Accuracy'] + df['Specificity'] + df['Precision'] + df['MCC'])
    df.to_csv(result_file, index=False)

    metrics = [(m, lambda a, b, c: 1) for m in [
        'MCC',
        'Specificity',
        'AUC',
        'Accuracy',
        'Precision',
        'Recall',
        'Mixed'
    ]]

    process_comparison_results(RESULT_DIR, metrics=metrics)
    process_comparison_results_plots(RESULT_DIR, metrics=metrics)
    process_comparison_result_winners(RESULT_DIR, metrics=metrics)
    process_times(RESULT_DIR)
    process_wilcoxon(classifiers, RESULT_DIR, metrics=metrics)
    process_parsing_of_rules(RESULT_DIR)
