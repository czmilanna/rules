import dataclasses
import re
from pathlib import Path

import pandas as pd
from cacp.util import to_latex


@dataclasses.dataclass
class AlgorithmRules:
    Algorithm: str
    Dataset: str
    number_of_characters: int = 0
    number_of_rules: int = 0
    number_of_attributes: int = 0
    number_of_unique_attributes: int = 0
    rules: list[str] = dataclasses.field(default_factory=list, repr=False)

    def calc_stats(self):
        self.number_of_characters = sum(len(r) for r in self.rules)
        self.number_of_rules = len(self.rules)
        attrs = []
        for r in self.rules:
            if 'THEN' in r:
                if_part, _ = r.split('THEN')
                if_part = if_part[2:]
                for expr in if_part.split(' AND '):
                    expr_split = re.split('>|<|=| is | in ', expr)
                    if '+' in expr:
                        n_expr = expr_split[0]
                        for e in n_expr.split('+'):
                            if '*' in e:
                                attr = e.split('*')[1].strip()
                                attrs.append(attr)
                    else:
                        if self.Algorithm == 'OIGA-C':
                            attr = expr_split[1].strip()
                        else:
                            attr = expr_split[0].strip()
                        attrs.append(attr)

        self.number_of_attributes = len(attrs)
        self.number_of_unique_attributes = len(set(attrs))


def process_1rc_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)
    for l in lines:
        if l.startswith('Rule'):
            _, rule = l.split(': ')
            rule, _ = rule.split("     ")
            rule: str = rule.replace("  ", " ")
            if_part, after = rule.split('THEN')
            if_part = if_part.replace(' in ', ' = ')
            label = after.split('->')[1].strip()
            metadata.rules.append(f'{if_part}THEN {label}')
    return metadata


def process_c45rulesc_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)
    prev_l = None
    for l in lines:
        if prev_l is not None:
            label = l.replace('output=', '').strip()
            if 'if(' in prev_l:
                _, right = prev_l.split('if(')
                left, *oth = right.split(')')
                args = left.replace("&&", 'AND')
                metadata.rules.append(f'IF {args} THEN {label}')
            elif prev_l.strip() == 'else':
                metadata.rules.append(f'ELSE {label}')

        prev_l = l
    return metadata


def process_dtgac_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)
    for l in lines:
        if l.startswith('Rule'):
            _, rule = l.split(': ')
            rule: str = rule.replace("  ", " ")
            if_part, after = rule.split('THEN')
            label = after.split('=')[1].split("(")[0].strip()
            metadata.rules.append(f'{if_part}THEN {label}')
    return metadata


def process_eachc_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)
    for l in lines:
        if l.startswith('Rule'):
            _, rule = l.split(': ')
            rule: str = rule.replace("  ", " ")
            if_part, after = rule.split('THEN')
            if_part = if_part.replace("IF  AND", "IF")
            label = after.split('->')[1].split("[")[0].strip()
            metadata.rules.append(f'{if_part}THEN {label}')
    return metadata


def process_gpr_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)
    for l in lines:
        if '|' in l:
            l = l.split('|')[0]
        metadata.rules.append(l)
    return metadata


def process_hiderc_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)
    if_parts = []
    prev_l = ''
    for l in lines:
        if '= [' in l:
            p = l.lstrip('if ').rstrip(' and')
            if_parts.append(p)
        if 'then' in prev_l and len(if_parts) > 0:
            label = l.split('=')[1].split('(')[0].strip()
            metadata.rules.append(f'IF {" AND ".join(if_parts)} THEN {label}')
            if_parts = []
        prev_l = l
    return metadata


def process_nslvc_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)
    if_parts = []
    start_reading_if_parts = False
    for l in lines:
        if l.strip() == 'IF':
            if_parts = []
            start_reading_if_parts = True
        elif l.startswith('THEN '):
            label = l.split(' IS ')[1].split(' W ')[0].strip()
            metadata.rules.append(f'IF {" AND ".join(if_parts)} THEN {label}')
        elif start_reading_if_parts:
            if_parts.append(l.strip())
    return metadata


def process_ocecc_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)

    for l in lines:
        if l.startswith('Rule'):
            _, rule_part, *o = l.split(':')
            if_part, then_part = rule_part.split('THEN')
            if_part = if_part.strip()
            label = then_part.split('=')[1].split('(')[0].strip()
            metadata.rules.append(f'{if_part} THEN {label}')
    return metadata


def process_oigac_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)

    for l in lines:
        l = l.strip()
        if l:
            metadata.rules.append(l)
    return metadata


def process_priglac_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)

    for l in lines:
        if " = [" in l:
            _, rule_part, label_part = l.split(':')
            rule = f'IF{rule_part} THEN{label_part}'
            metadata.rules.append(rule)
    return metadata


@dataclasses.dataclass
class Node:
    value: str
    children: list['Node']


def traverse(nodes: list[Node]):
    paths = []

    def _traverse(node: Node, path: list[str]):
        path.append(node.value)
        if len(node.children) == 0:
            paths.append(path.copy())
            path.pop()
        else:
            for child in node.children:
                _traverse(child, path)
            path.pop()

    for n in nodes:
        _traverse(n, [])

    return paths


def process_c45c_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)

    tree_lines = [l for l in lines if not l.startswith('@')]

    def parse_value(s: str):
        return s.split('(')[1].split(')')[0].strip()

    def parse_nodes(sub_lines: list[str], depth: int = 0):
        if len(sub_lines) == 3:
            value = parse_value(sub_lines[0])
            label = sub_lines[2].split('=')[1].replace('"', "").strip()
            return [Node(f'THEN {label}', [])]

        nodes_values = []
        nodes_starts = []
        nodes_ends = []
        pref = '\t' * depth
        for i, l in enumerate(sub_lines):
            if l.startswith(pref + 'if (') or l.startswith(pref + 'elseif ('):
                value = parse_value(l)
                nodes_values.append(value)
                nodes_starts.append(i)
            if l.startswith(pref + "}"):
                nodes_ends.append(i)

        nodes = []
        for v, s, e in zip(nodes_values, nodes_starts, nodes_ends):
            nodes.append(Node(v, parse_nodes(sub_lines[s: e], depth + 1)))

        return nodes

    nodes = parse_nodes(tree_lines)

    for p in traverse(nodes):
        metadata.rules.append(f'IF {" AND ".join(p[:-1])} {p[-1]}')

    return metadata


def process_dtoblique_file(algorithm: str, dataset: str, lines: list[str]) -> AlgorithmRules:
    metadata = AlgorithmRules(algorithm, dataset)

    tree_lines = [l for l in lines if not l.startswith('Accuracy ')]

    def parse_value(s: str):
        return s.split('(')[1].split(')')[0].strip()

    def parse_nodes(sub_lines: list[str], depth: int = 0):
        if len(sub_lines) == 1:
            label = sub_lines[0].split('=')[1].split('(')[0].strip()
            return [Node(f'THEN {label}', [])]
        pref = '\t' * depth
        sub_pref = '\t' * (depth + 1)
        node_value = parse_value(sub_lines[0])
        if_lines = []
        else_lines = []
        found_else = False
        for i, l in enumerate(sub_lines):
            if l.startswith(pref + 'else{'):
                found_else = True
                continue
            if l.startswith(sub_pref):
                if found_else:
                    else_lines.append(l)
                else:
                    if_lines.append(l)

        level_nodes = []
        level_nodes.append(
            Node(node_value, parse_nodes(if_lines, depth + 1))
        )
        level_nodes.append(
            Node(node_value.replace('>= 0', '< 0'), parse_nodes(else_lines, depth + 1))
        )

        return level_nodes

    nodes = parse_nodes(tree_lines)

    for p in traverse(nodes):
        metadata.rules.append(f'IF {" AND ".join(p[:-1])} {p[-1]}')

    return metadata


file_processors = {
    "1R-C": process_1rc_file,
    "C45-C": process_c45c_file,
    "C45Rules-C": process_c45rulesc_file,
    "C45RulesSA-C": process_c45rulesc_file,  # same processor as C45Rules-C works
    'DT_GA-C': process_dtgac_file,
    'DT_Oblique-C': process_dtoblique_file,
    'EACH-C': process_eachc_file,
    'GPR': process_gpr_file,
    'Hider-C': process_hiderc_file,
    'NSLV-C': process_nslvc_file,
    'OCEC-C': process_ocecc_file,
    'OIGA-C': process_oigac_file,
    'PGIRLA-C': process_priglac_file,
    'Ripper-C': process_c45rulesc_file,  # same processor as C45Rules-C works
    'SLAVE2-C': process_nslvc_file,  # same processor as NSLV-C works
    'SLAVEv0-C': process_nslvc_file,  # same processor as NSLV-C works
}


def process_file(dataset: str, algorithm: str, path: Path) -> AlgorithmRules:
    processor = file_processors.get(algorithm)
    if processor is not None:
        with path.open('r') as f:
            lines = [l.rstrip() for l in f.readlines() if l.strip()]
            return processor(algorithm, dataset, lines)


def process_parsing_of_rules(
        result_dir: Path,
):
    parsed_result_dir = result_dir.joinpath('rules_parsed')
    records = []
    for path in result_dir.joinpath('rules').glob('*/*/result.txt'):
        dataset = path.parts[-3]
        algorithm = path.parts[-2]
        metadata = process_file(dataset, algorithm, path)
        if metadata is not None:
            out_dir = parsed_result_dir.joinpath(dataset).joinpath(algorithm)
            out_dir.mkdir(exist_ok=True, parents=True)
            with out_dir.joinpath('rules.txt').open('w') as f:
                f.write('\n'.join(metadata.rules))
                metadata.calc_stats()
                meta_dict = metadata.__dict__
                del meta_dict['rules']
                records.append(meta_dict)

    records_df = pd.DataFrame(records)
    records_df.index += 1
    records_df.to_csv(parsed_result_dir.joinpath('rules.csv'))

    df = records_df.groupby(
        ['Algorithm']
    ).mean().rename(columns={
        'number_of_characters': 'ANOC',
        'number_of_rules': 'ANR',
        'number_of_attributes': 'ANA',
        'number_of_unique_attributes': 'ANUA',
    }).sort_values(by=['ANOC', 'ANR', 'ANA', 'ANUA'], ascending=True)

    df.reset_index(inplace=True)
    df.index += 1
    df.to_csv(parsed_result_dir.joinpath('rules_result.csv'))

    f = parsed_result_dir.joinpath(f'rules_result.tex').open('w')
    f.write(
        to_latex(
            df,
            caption=f"Rules statistics",
            label=f'tab:rules_result',
        )
    )
    print(df)


if __name__ == '__main__':
    process_parsing_of_rules(Path(__file__).parent.joinpath('result'))
