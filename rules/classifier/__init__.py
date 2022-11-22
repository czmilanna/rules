import subprocess
import typing
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import typing_extensions

CLASSIFIER_DIR = Path(__file__).parent
CLASSIFIER_METHODS_DIR = CLASSIFIER_DIR.joinpath("keel_methods")

AVAILABLE_CLASSIFIERS_NAMES = typing_extensions.Literal[
    'NSLV-C', 'RISE-C', 'PGIRLA-C', 'PDFC-C', 'SLAVEv0-C', 'EHS_CHC-C',
    'MPLCS-C', 'C45Rules-C', 'OCEC-C', 'CART-C', 'Ripper-C', 'BNGE-C',
    'Slipper-C', 'EACH-C', 'C45RulesSA-C', 'C45-C', 'Hider-C', 'FURIA-C',
    'DT_GA-C', 'DT_Oblique-C', '1R-C', 'AdaBoost.NC-C', 'INNER-C', 'PART-C',
    'SIA-C', 'OIGA-C', 'SLAVE2-C']

options_type = typing.Union[typing.List[str], None]
numeric_type = typing.Union[int, float, None]
value_type = typing.Union[str, numeric_type]


class ClassifierParameter:
    def __init__(self, name: str, type: str, options: options_type, min: numeric_type, max: numeric_type,
                 default: numeric_type):
        self.name = name
        self.type = type
        self.options = options
        self.min = min
        self.max = max
        self.default = default

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items() if item[1] is not None)
        )

    def __repr__(self):
        return self.__str__()

    def validate(self, value: value_type):
        if self.type == "integer" and type(value) != int:
            raise ValueError(
                f"Parameter {self.name} has invalid value {value}, should have int type"
            )
        elif self.type == "real" and type(value) != float:
            raise ValueError(
                f"Parameter {self.name} has invalid value {value}, should have float type"
            )
        elif self.type == "text" and type(value) != str:
            raise ValueError(
                f"Parameter {self.name} has invalid value {value}, should have str type"
            )
        elif self.type == "list" and value not in self.options:
            raise ValueError(
                f"Parameter {self.name} has invalid value {value}, should have one of {self.options}"
            )
        elif self.min is not None and self.max is not None:
            if value < self.min or value > self.max:
                raise ValueError(
                    f"Parameter {self.name} has invalid value {value}, should be between [{self.min}, {self.max}]"
                )


class Classifier:
    def __init__(self, name: str, parameters_overrides: dict = None, seed: int = 1):
        self.name = name
        self.parameters_overrides = parameters_overrides or {}
        self._parse_data_from_list_file()
        self._parse_data_from_metadata_file()
        self._validate_parameters_overrides()
        self.seed = seed if self.accepts_seed else None
        self.config_file_path: Path = Path()
        self.train_result_file_path: Path = Path()
        self.test_result_file_path: Path = Path()
        self.additional_result_file_path: Path = Path()

    def _parse_data_from_list_file(self):
        methods_file_path = CLASSIFIER_DIR.joinpath("keel_methods.xml")
        root = ET.parse(methods_file_path).getroot()
        for method_tag in root.findall('method'):
            name = method_tag.find("name").text
            if name == self.name:
                self.name = name
                self.jar_file_name = method_tag.find("jar_file").text
                self.family = method_tag.find("family").text
                input_tag = method_tag.find("input")

                def get_input_value(input_name: str):
                    if input_tag is not None:
                        input_value_tag = input_tag.find(input_name)
                        return input_value_tag.text == "Yes" if input_value_tag is not None else False
                    return False

                self.continuous = get_input_value("continuous")
                self.integer = get_input_value("integer")
                self.nominal = get_input_value("nominal")
                self.missing = get_input_value("missing")
                self.imprecise = get_input_value("imprecise")
                self.multi_class = get_input_value("multiclass")
                self.multi_output = get_input_value("multioutput")
                self.multi_instance = get_input_value("multiinstance")

    def _parse_data_from_metadata_file(self):
        root = ET.parse(self.xml_file_path).getroot()
        self.full_name = root.find("name").text
        self.accepts_seed = root.find("seed").text == "1"
        self.parameters: typing.List[ClassifierParameter] = []
        self.parameters_by_name: typing.Dict[str, ClassifierParameter] = {}
        for parameter_tag in root.findall('parameter'):
            parameter_name = parameter_tag.find("name").text
            parameter_type = parameter_tag.find("type").text
            parameter_default = parameter_tag.find("default").text
            parameter_options = None
            parameter_min = None
            parameter_max = None
            parameter_domain_tag = parameter_tag.find("domain")
            if parameter_type == "list":
                parameter_options = [o.text for o in parameter_domain_tag.findall('item')]
            elif parameter_type == "integer" and parameter_domain_tag:
                parameter_min = int(parameter_domain_tag.find("lowerB").text)
                parameter_max = int(parameter_domain_tag.find("upperB").text)
                parameter_default = int(parameter_default)
            elif parameter_type == "real" and parameter_domain_tag:
                parameter_min = float(parameter_domain_tag.find("lowerB").text)
                parameter_max = float(parameter_domain_tag.find("upperB").text)
                parameter_default = float(parameter_default)
                if parameter_default.is_integer():
                    parameter_default = int(parameter_default)

            parameter = ClassifierParameter(
                name=parameter_name,
                type=parameter_type,
                options=parameter_options,
                min=parameter_min,
                max=parameter_max,
                default=parameter_default
            )
            self.parameters.append(
                parameter
            )
            self.parameters_by_name[parameter.name] = parameter

    def _validate_parameters_overrides(self):
        for parameter_name, parameter_value in self.parameters_overrides.items():
            parameter = self.parameters_by_name.get(parameter_name)
            if parameter is None:
                raise ValueError(f"Parameter {parameter_name} is not valid for Classifier {self.name}")
            else:
                parameter.validate(parameter_value)

    @property
    def jar_file_path(self) -> Path:
        return CLASSIFIER_METHODS_DIR.joinpath(self.jar_file_name)

    @property
    def xml_file_path(self) -> Path:
        return CLASSIFIER_METHODS_DIR.joinpath(f"{self.name}.xml")

    def __str__(self):
        ignored = ['parameters_by_name', 'accepts_seed']
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join(f'{name}={value}' for name, value
                      in vars(self).items() if value is not None and name not in ignored)
        )

    def __repr__(self):
        return self.__str__()

    def fit_predict(self,
                    result_file_path: Path,
                    train_file_path: Path, test_file_path: Path,
                    ):
        result_file_path.mkdir(exist_ok=True, parents=True)
        self.config_file_path = result_file_path.joinpath("config.txt")
        self.train_result_file_path = result_file_path.joinpath("result.tra")
        self.test_result_file_path = result_file_path.joinpath("result.tst")
        self.additional_result_file_path = result_file_path.joinpath("result.txt")

        def path_to_keel_format(path: Path):
            return str(path).replace('\\', '/')

        lines = [
            f"algorithm = {self.full_name}",
            f'inputData = "{path_to_keel_format(train_file_path)}" "{path_to_keel_format(train_file_path)}" "{path_to_keel_format(test_file_path)}"',
            f'outputData = "{path_to_keel_format(self.train_result_file_path)}" '
            f'"{path_to_keel_format(self.test_result_file_path)}" '
            f'"{path_to_keel_format(self.additional_result_file_path)}"',
            ''
        ]

        if self.seed is not None:
            lines.append(f'seed = {self.seed}')
        for parameter in self.parameters:
            value = parameter.default
            value_override = self.parameters_overrides.get(parameter.name)
            if value_override is not None:
                value = value_override
            lines.append(f'{parameter.name} = {value}')
        lines.append('')
        content = "\n".join(lines)

        with self.config_file_path.open("wb") as f:
            f.write(bytes(content, "UTF-8"))

        run_command = f"java -jar {path_to_keel_format(self.jar_file_path)} {path_to_keel_format(self.config_file_path)}"
        process = subprocess.Popen(run_command, stdout=subprocess.DEVNULL)
        timeout = None
        # timeout = 30 # todo remove after working testing
        process.wait(timeout=timeout)
        if process.returncode != 0:
            raise RuntimeError(f"Error while running classifier {self.name} for config file {self.config_file_path}")

        return self.classification_test_results()

    def classification_test_results(self):
        y_true = []
        y_pred = []
        with self.test_result_file_path.open("r") as f:
            for line in f.readlines():
                if '@' not in line:
                    true, pred = line.split()
                    y_true.append(true)
                    y_pred.append(pred)
        return np.array(y_true), np.array(y_pred)

    @property
    def rules(self) -> list[str]:
        return []


def all_classifiers() -> typing.List[Classifier]:
    """
    Gets all available classifiers

    :return: all classifiers
    """
    return [
        Classifier(name) for name in typing_extensions.get_args(AVAILABLE_CLASSIFIERS_NAMES)
    ]


def all_classifiers_with_nice_rules():
    names = ['1R-C', 'C45-C', 'C45Rules-C', 'C45RulesSA-C', 'DT_GA-C',
             'DT_Oblique-C', 'EACH-C', 'Hider-C', 'NSLV-C', 'OCEC-C',
             'OIGA-C', 'PGIRLA-C', 'Ripper-C', 'SLAVE2-C', 'SLAVEv0-C']

    return [
        Classifier(name) for name in names
    ]
