from pathlib import Path

import pytest

from research.parse_rules import AlgorithmRules


@pytest.fixture()
def parsed_rules_path() -> Path:
    return Path(__file__).parent.parent.parent.joinpath('research').joinpath('result').joinpath('rules_parsed')


def test_rules_calc_stats_gpr(parsed_rules_path: Path):
    metadata = AlgorithmRules('GPR', 'wisconsin')
    with parsed_rules_path.joinpath(metadata.Dataset, metadata.Algorithm, 'rules.txt').open('r') as f:
        metadata.rules = f.readlines()
    metadata.calc_stats()
    assert metadata.number_of_rules == 5
    assert metadata.number_of_characters == 252
    assert metadata.number_of_attributes == 9
    assert metadata.number_of_unique_attributes == 6


def test_rules_calc_stats_c45c(parsed_rules_path: Path):
    metadata = AlgorithmRules('C45-C', 'wisconsin')
    with parsed_rules_path.joinpath(metadata.Dataset, metadata.Algorithm, 'rules.txt').open('r') as f:
        metadata.rules = f.readlines()
    metadata.calc_stats()
    assert metadata.number_of_rules == 13
    assert metadata.number_of_characters == 1458
    assert metadata.number_of_attributes == 51
    assert metadata.number_of_unique_attributes == 7


def test_rules_calc_stats_eachc(parsed_rules_path: Path):
    metadata = AlgorithmRules('EACH-C', 'wisconsin')
    with parsed_rules_path.joinpath(metadata.Dataset, metadata.Algorithm, 'rules.txt').open('r') as f:
        metadata.rules = f.readlines()
    metadata.calc_stats()
    assert metadata.number_of_rules == 2
    assert metadata.number_of_characters == 595
    assert metadata.number_of_attributes == 18
    assert metadata.number_of_unique_attributes == 9


def test_rules_calc_stats_hiderc(parsed_rules_path: Path):
    metadata = AlgorithmRules('Hider-C', 'wisconsin')
    with parsed_rules_path.joinpath(metadata.Dataset, metadata.Algorithm, 'rules.txt').open('r') as f:
        metadata.rules = f.readlines()
    metadata.calc_stats()
    assert metadata.number_of_rules == 1
    assert metadata.number_of_characters == 115
    assert metadata.number_of_attributes == 4
    assert metadata.number_of_unique_attributes == 4


def test_rules_calc_stats_dtobliquec(parsed_rules_path: Path):
    metadata = AlgorithmRules('DT_Oblique-C', 'wisconsin')
    with parsed_rules_path.joinpath(metadata.Dataset, metadata.Algorithm, 'rules.txt').open('r') as f:
        metadata.rules = f.readlines()
    metadata.calc_stats()
    assert metadata.number_of_rules == 27
    assert metadata.number_of_characters == 20647
    assert metadata.number_of_attributes == 584
    assert metadata.number_of_unique_attributes == 9


def test_rules_calc_stats_oigac(parsed_rules_path: Path):
    metadata = AlgorithmRules('OIGA-C', 'wisconsin')
    with parsed_rules_path.joinpath(metadata.Dataset, metadata.Algorithm, 'rules.txt').open('r') as f:
        metadata.rules = f.readlines()
    metadata.calc_stats()
    assert metadata.number_of_rules == 30
    assert metadata.number_of_characters == 15593
    assert metadata.number_of_attributes == 270
    assert metadata.number_of_unique_attributes == 9
