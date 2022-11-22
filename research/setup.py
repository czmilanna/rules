from rules.classifier import Classifier
from rules.classifier.gpr import GPRKeelCompatible

names = ['1R-C', 'C45-C', 'C45Rules-C', 'C45RulesSA-C', 'DT_GA-C',
         'DT_Oblique-C', 'EACH-C', 'Hider-C', 'NSLV-C', 'OCEC-C',
         'OIGA-C', 'PGIRLA-C', 'Ripper-C', 'SLAVE2-C', 'SLAVEv0-C']


def get_classifier_factory(name):
    return lambda: Classifier(name)


classifiers = [
                  (name, get_classifier_factory(name)) for name in names

              ] + [('GPR', lambda: GPRKeelCompatible())]
