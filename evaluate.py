import argparse
import json
import Evaluator
from PredictorFactory import PredictorFactory
import importlib

def evaluate(config):
    with open(config) as file:
        data=json.load(file)


    # Crear el readDataset correcto
    readDataset=data['readDataset']
    path = data['pathDataset']
    pathLabels = data['pathLabels']

    classReadDataset__ = getattr(importlib.import_module(readDataset), readDataset)
    rd=classReadDataset__()

    # Modificar los parámetros del constructor
    evaluator = Evaluator.Evaluator(rd, path,
                                    pathLabels)

    # Añadimos las medidas
    measures=data['measures']
    for measure in measures:
        evaluator.addMeasure(measure)

    # Añadimos los constructores
    m = PredictorFactory()
    predictors=data['predictors']
    for predictor in predictors:
        # Crear los modelos con los nombres de los predictores (framework, model)
        model = m.getPredictor(predictor['framework'], predictor['model'])
        evaluator.addPredictor(model)

    evaluator.evaluate()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Configuration file")
    args=vars(parser.parse_args())
    evaluate(args["config"])
