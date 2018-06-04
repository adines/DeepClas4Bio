from ModelFactory import ModelFactory
import importlib

class PredictorFactory:
    def getPredictor(self,framework,model):
        modelFact=ModelFactory()
        m=modelFact.getModel(framework,model)
        class_name=framework+'Predictor'
        class__ = getattr(importlib.import_module(framework + '.' + class_name), class_name)
        return class__(m)