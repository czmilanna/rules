import dataclasses
import os
import time
import typing
from pathlib import Path

import numpy as np
import pandas as pd
from cacp.dataset import AVAILABLE_N_FOLDS
from cacp.util import auc_score
from joblib import delayed, Parallel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from research import DATA_DIR, RESULT_DIR
from rules.classifier import Classifier


def process_comparison_of_rules(
        classifiers: list[tuple[str, typing.Callable[[], Classifier]]],
        data_dir: Path,
        result_dir: Path,
):
    """
    Runs comparison for provided datasets and classifiers.

    :param classifiers: classifiers collection
    :param result_dir: results directory

    """

    dataset_paths = [p.joinpath(f'{p.name}.dat') for p in data_dir.glob('*')]

    for dataset_path in dataset_paths:
        print(dataset_path.stem)
        for classifier_name, classifier_factory in classifiers:
            cls = classifier_factory()
            r_dir = result_dir.joinpath(dataset_path.stem).joinpath(classifier_name)
            y_true, y_pred = cls.fit_predict(r_dir, dataset_path, dataset_path)
            labels = np.unique(y_true)
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
                # 'auc': auc_score(y_true, y_pred, average='weighted', multi_class='ovo', labels=labels),
            }
            with r_dir.joinpath('metrics.txt').open('w') as f:
                metrics_strings = [f"{k}: {v}" for k, v in metrics.items()]
                f.write("\n".join(metrics_strings))
