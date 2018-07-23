import DatasetManager
import importlib
import json

class Evaluator:
    def __init__(self,readDataset,path,pathLabels,batch=64):
        self.readDataset=readDataset
        [self.images,self.labels]=readDataset.readDataset(path,pathLabels)
        self.predictors=[]
        self.measures=[]
        self.batch=batch
        if len(list(set(self.labels))) <2:
            raise ValueError('Only one class detected. Review your dataset.')
        elif len(list(set(self.labels))) == 2:
            self.binary=True
        else:
            self.binary=False


    def addPredictor(self,predictor):
        self.predictors.append(predictor)

    def addMeasure(self, measure):
        self.measures.append(measure)

    def evaluate(self):
        result={}
        for predictor in self.predictors:
            if hasattr(predictor.model,'name'):
                predictorName=predictor.model.name
            else:
                predictorName='DL4J'+predictor.model
            dataManager=DatasetManager.DatasetManager(self.images, batch=self.batch)
            predictions=[]
            while(dataManager.hasNextBach()):
                batchImages=dataManager.nextBatch()
                prediction=predictor.predictBatch(batchImages)
                predictions+=prediction
            resultMeasure={}
            # Mirar si las medidas son adecuadas, binarias o no
            for measure in self.measures:
                measureMethod_ = getattr(importlib.import_module('Measures'), measure)
                r = measureMethod_(predictions, self.labels)
                resultMeasure[measure]=r
            result[predictorName]=resultMeasure
        with open('data.json', 'w') as f:
            json.dump(result, f)