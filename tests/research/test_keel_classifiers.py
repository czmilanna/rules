from pathlib import Path

from sklearn.metrics import accuracy_score

from rules.classifier import all_classifiers_with_nice_rules
from rules.classifier.gpr import GPRKeelCompatible


def test_keel_classifiers(data_path: Path, result_path: Path):
    wisconsin_path = data_path.joinpath('wisconsin')
    for classifier in all_classifiers_with_nice_rules() + [GPRKeelCompatible()]:
        result_path = result_path.joinpath(classifier.name).absolute()
        y_true, y_pred = classifier.fit_predict(
            result_path,
            wisconsin_path.joinpath("wisconsin-10-1tra.dat").absolute(),
            wisconsin_path.joinpath("wisconsin-10-1tst.dat").absolute(),
        )
        print(classifier.name, accuracy_score(y_true, y_pred))
