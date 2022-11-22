import dataclasses
import os
import time
import typing
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from cacp.dataset import AVAILABLE_N_FOLDS
from cacp.util import auc_score
from joblib import delayed, Parallel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from tqdm import tqdm

from research import DATA_DIR, RESULT_DIR
from rules.classifier import Classifier


@dataclasses.dataclass
class Fold:
    index: int
    train_data_path: Path
    test_data_path: Path


def process_comparison_single(classifier_factory: typing.Callable[[], Classifier],
                              classifier_name: str,
                              dataset: str,
                              fold: Fold) -> dict:
    """
    Runs comparison on single classifier and dataset.

    :param classifier_factory: classifier factory
    :param classifier_name: classifier name
    :param dataset: single dataset
    :param fold: fold data
    :return: dictionary of calculated metrics and metadata

    """
    cls = classifier_factory()
    train_start_time = time.time()

    result_dir = RESULT_DIR.joinpath('runs').joinpath(dataset).joinpath(classifier_name).joinpath(str(fold.index))
    y_true, y_pred = cls.fit_predict(result_dir, fold.train_data_path, fold.test_data_path)
    y_pred[y_pred == '?'] = y_true[0]
    labels = np.unique(np.concatenate((y_true, y_pred), axis=None))

    train_time = (time.time() - train_start_time)

    arr = confusion_matrix(y_true, y_pred, labels=labels)
    # print(arr, dataset, fold.index, labels)
    TN, FP, FN, TP = arr.ravel()

    def mcc():
        try:
            return float((TP * TN) - (FP * FN)) / (sqrt(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))))
        except:
            return 0.

    def ts():
        try:
            return TP / (TP + FN + FP)
        except:
            return 0.

    def specificity():
        try:
            return TN / (TN + FP)
        except:
            return 0.

    return {
        'Dataset': dataset,
        'Algorithm': classifier_name,
        'Number of classes': -1,
        'Train size': -1,
        'Test size': -1,
        'CV index': fold.index,
        'MCC': mcc(),
        'TS': ts(),
        'Specificity': specificity(),
        'AUC': auc_score(y_true, y_pred, average='weighted', multi_class='ovo', labels=labels),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
        'F1': f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TP': TP,
        'Train time [s]': train_time,
        'Prediction time [s]': -1  # can not be calculated
    }


def _get_folds(data_path: Path, dataset: str, n_folds: int = 10, dob_scv: bool = True):
    data_path = data_path.joinpath(dataset)
    if dob_scv:
        data_name = f'{dataset}-{n_folds}dobscv'
    else:
        data_name = f'{dataset}-{n_folds}'

    for fold_index in range(1, n_folds + 1):
        train_data_path = data_path.joinpath(f'{data_name}-{fold_index}tra.dat')
        test_data_path = data_path.joinpath(f'{data_name}-{fold_index}tst.dat')
        yield Fold(fold_index, train_data_path, test_data_path)


def process_comparison(
        classifiers: list[tuple[str, typing.Callable[[], Classifier]]],
        data_path: Path,
        result_dir: Path,
        n_jobs: int = 8,
        n_folds: AVAILABLE_N_FOLDS = 10,
        dob_scv: bool = True
):
    """
    Runs comparison for provided datasets and classifiers.

    :param classifiers: classifiers collection
    :param result_dir: results directory
    :param n_folds: number of folds {5,10}
    :param dob_scv: if folds distribution optimally balanced stratified cross-validation (DOB-SCV) should be used

    """
    count = 0
    records = []
    df = None

    dataset_names = [p.name for p in data_path.glob('*')]

    with tqdm(total=len(dataset_names) * n_folds, desc='Processing comparison', unit='fold') as pbar:
        for dataset_idx, dataset in enumerate(dataset_names):
            for fold_idx, fold in enumerate(_get_folds(data_path, dataset, n_folds=n_folds, dob_scv=dob_scv)):
                rows = Parallel(n_jobs=n_jobs)(
                    delayed(process_comparison_single)(c, c_n, dataset, fold) for c_n, c in classifiers
                )
                records.extend(rows)
                pbar.update(1)

            df = pd.DataFrame(records)
            df = df.sort_values(by=['Dataset', 'Algorithm', 'CV index'])
            count += 1
            df.to_csv(result_dir.joinpath(f'comparison_{count}.csv'), index=False)
            if count > 1:
                prev_file = result_dir.joinpath(f'comparison_{count - 1}.csv')
                if os.path.isfile(prev_file):
                    os.remove(prev_file)

    if df is not None:
        prev_file = result_dir.joinpath(f'comparison_{count}.csv')
        if os.path.isfile(prev_file):
            os.remove(prev_file)
        df.to_csv(result_dir.joinpath('comparison.csv'), index=False)
