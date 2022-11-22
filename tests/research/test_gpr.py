from pathlib import Path

from sklearn.metrics import accuracy_score

from rules.classifier.gpr import GPRKeelCompatible


def test_keel_gpr(data_path: Path, result_path: Path):
    classifier = GPRKeelCompatible()
    result_path = result_path.joinpath(classifier.name).absolute()
    ds_name = "type1diabetes2"
    ds_path = data_path.joinpath(ds_name)
    y_true, y_pred = classifier.fit_predict(
        result_path,
        ds_path.joinpath(f"{ds_name}-10-1tra.dat").absolute(),
        ds_path.joinpath(f"{ds_name}-10-1tst.dat").absolute(),
    )
    print(classifier.name, accuracy_score(y_true, y_pred), classifier.cls.rules)
