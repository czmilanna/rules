import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture()
def data_path() -> Path:
    return Path(__file__).parent.parent.joinpath('data')


@pytest.fixture()
def result_path() -> Path:
    return Path(__file__).parent.joinpath('result')


@pytest.fixture()
def strange_result_files_to_delete() -> list[Path]:
    return [
        Path(__file__).parent.joinpath('arbol.txt'),
        Path(__file__).parent.joinpath('classifier').joinpath('arbol.txt')
    ]


@pytest.fixture(autouse=True)
def run_before_and_after_tests(result_path, strange_result_files_to_delete):
    """Fixture to execute asserts before and after a test is run"""
    # Setup

    # Run
    yield

    # Teardown
    shutil.rmtree(result_path, ignore_errors=True)
    for f in strange_result_files_to_delete:
        if f.exists():
            os.remove(f)
