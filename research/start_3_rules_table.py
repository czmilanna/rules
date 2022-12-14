import pandas as pd
from cacp.plot import process_comparison_results_plots
from cacp.result import process_comparison_results
from cacp.time import process_times
from cacp.util import seed_everything, to_latex
from cacp.wilcoxon import process_wilcoxon
from cacp.winner import process_comparison_result_winners

from research import RESULT_DIR
from research.parse_rules import process_parsing_of_rules
from research.setup import classifiers

if __name__ == '__main__':
    seed_everything(1)
    records = []
    for r_path in  RESULT_DIR.joinpath("rules_parsed", "type1diabetes2").glob("*/*.txt"):
        with r_path.open('r') as f:
            algorithm = r_path.parent.name
            lines = f.readlines()
            number_of_rules  = len(lines)
            rule_length = sum(len(l) for l in lines)
            rule_string = '\\\\'.join(lines)
            if len(rule_string) > 180:
                rule_string = rule_string[:177] + "..."
            rule_string = rule_string.replace('{', '\\{')
            rule_string = rule_string.replace('}', '\\}')
            rule_string = rule_string.replace('>', '\\textgreater{}')
            rule_string = rule_string.replace('<', '\\textless{}')
            rule_string = "\\begin{tabular}[c]{m{0.5\\textwidth}}" + rule_string + "\\end{tabular}"

            record = {
                'Algorithm': algorithm,
                'Rules': rule_string,
                'Number': number_of_rules,
                'Length': rule_length,
            }
            records.append(record)

    df = pd.DataFrame(records)
    df.index += 1
    f = RESULT_DIR.joinpath(f'example_rules.tex').open('w')
    f.write(
        to_latex(
            df,
            caption=f'Example of "if-then" fuzzy rules generated by rule-based fuzzy logic classifiers on the real Type 1 Diabetes dataset.',
            label=f'tab:example_rules',
        )
    )

