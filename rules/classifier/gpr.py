from pathlib import Path

import numpy as np
import pandas as pd
from gpr_algorithm.algorithm import GPR
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from rules.classifier import Classifier


class GPRKeelCompatible(Classifier):

    def __init__(self, n_populations=500, n_generations=10):
        self.n_populations = n_populations
        self.n_generations = n_generations
        self.cls = None
        self._instances = 0
        self._features = 0
        self._classes = 0
        self._output = 'Class'
        self._origin = ''
        self._attributes: dict[str, str] = {}
        self.accepts_seed = True
        self.feature_names = []
        super().__init__('GPR')

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__, "TODO"
        )

    def __repr__(self):
        return self.__str__()

    def _parse_data_from_list_file(self):
        pass

    def _parse_data_from_metadata_file(self):
        pass

    def _validate_parameters_overrides(self):
        pass

    def fit_predict(self,
                    result_file_path: Path,
                    train_file_path: Path, test_file_path: Path,
                    ):
        self._load_description(train_file_path)

        self.cls = GPR(feature_names=self.feature_names,
                       n_populations=self.n_populations, n_generations=self.n_generations,
                       verbose=False)

        self.cls.feature_names = list(self._attributes.keys())

        x_tra, y_tra = self._load_data(train_file_path)
        x_tst, y_tst = self._load_data(test_file_path)

        # normalize
        x_tra_len = len(x_tra)
        x = np.concatenate([x_tra.astype(float), x_tst.astype(float)])
        min_max_scaler = preprocessing.MinMaxScaler()
        x = min_max_scaler.fit_transform(x)
        x_train, x_test = x[:x_tra_len], x[x_tra_len:]

        self.cls.fit(x_train, y_tra)

        y_pred = self.cls.predict(x_test)

        result_file_path.mkdir(exist_ok=True, parents=True)

        with result_file_path.joinpath("result.txt").open('w') as f:
            f.write('\n'.join(self.cls.rules))

        return y_tst, y_pred

    def _load_description(self, path: Path):
        attributes_names = []
        attributes_types_names = []
        inputs = []
        output_name = 'Class'
        # KEEL descriptions files contain latin1 chars
        with path.open('r', encoding='latin1') as file:
            for line in file:
                if '@attribute' in line or '@Attribute' in line:
                    if '{' in line:
                        attr_name = line.split('{')[0].split()[1]
                        attr_type = 'category'
                    else:
                        s = line.split()[1:]
                        attr_name = s[0].strip()
                        attr_type = s[1].split('[')[0].strip()
                    attributes_names.append(attr_name)
                    attributes_types_names.append(attr_type)
                if '@input' in line:
                    inputs.append(line.split()[1:])
                elif '@output' in line:
                    output_name = line.split()[1]
                elif 'Origin.' in line:
                    self._origin = line.split('Origin.')[1].strip()
                elif 'Features.' in line:
                    self._features = int(line.split('Features.')[1].strip())
                elif 'Classes.' in line:
                    self._classes = int(line.split('Classes.')[1].strip())
                elif 'Instances.' in line:
                    self._instances = int(line.split('Instances.')[1].split()[0].strip())
                elif '@data' in line:
                    break

        self._attributes = {n: t for n, t in zip(attributes_names, attributes_types_names)}
        self.feature_names = [a for a in attributes_names if a != output_name]
        self._output_name = output_name

    def _load_data(self, path: Path, categorical_to_numerical: bool = True) -> tuple[np.ndarray, np.ndarray]:
        skip_rows = 4 + len(self._attributes)
        df = pd.read_csv(path, skiprows=skip_rows, names=self._attributes.keys(), na_values='?')
        if categorical_to_numerical:
            for attr_name, attr_type_name in self._attributes.items():
                if attr_type_name == 'category':
                    df[attr_name] = df[attr_name].astype('category').cat.codes.values

        y = df[self._output_name].values
        del df[self._output_name]
        x = df.values
        return x, y
