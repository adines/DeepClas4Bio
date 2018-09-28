import argparse
import json
import Evaluator
from PredictorFactory import PredictorFactory
import importlib

def evaluate(config):
    with open(config) as file:
        data=json.load(file)


    readDataset=data['readDataset']
    path = data['pathDataset']
    pathLabels = data['pathLabels']

    classReadDataset__ = getattr(importlib.import_module(readDataset), readDataset)
    rd=classReadDataset__()

    evaluator = Evaluator.Evaluator(rd, path,
                                    pathLabels)

    # Anadimos las medidas
    measures=data['measures']
    for measure in measures:
        evaluator.addMeasure(measure)

    # Anadimos los constructores
    m = PredictorFactory()
    predictors=data['predictors']
    for predictor in predictors:

        model = m.getPredictor(predictor['framework'], predictor['model'])
        evaluator.addPredictor(model)

    evaluator.evaluate()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Configuration file")
    args=vars(parser.parse_args())
    evaluate(args["config"])
