from deepclas4bio import Evaluator
from deepclas4bio.PredictorFactory import PredictorFactory
import importlib

def evaluate(readDataset,path,pathLabels,measures,predictors):

    classReadDataset__ = getattr(importlib.import_module('deepclas4bio.'+readDataset), readDataset)
    rd=classReadDataset__()

    evaluator = Evaluator.Evaluator(rd, path,
                                    pathLabels)

    # Anadimos las medidas
    for measure in measures:
        evaluator.addMeasure(measure)

    # Anadimos los constructores
    m = PredictorFactory()
    for predictor in predictors:

        model = m.getPredictor(predictor['framework'], predictor['model'])
        evaluator.addPredictor(model)

    return evaluator.evaluate()
