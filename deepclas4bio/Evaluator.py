from deepclas4bio import DatasetManager, Measures
import importlib


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
            predictorName=predictor.model.name
            dataManager= DatasetManager.DatasetManager(self.images, batch=self.batch)
            predictions=[]
            while(dataManager.hasNextBach()):
                batchImages=dataManager.nextBatch()
                prediction=predictor.predictBatch(batchImages)
                predictions+=prediction
            resultMeasure={}
            # Mirar si las medidas son adecuadas, binarias o no
            if self.binary:
                allowedMeasures= Measures.binaryMeasures
            else:
                allowedMeasures= Measures.noBinaryMeasures
            for measure in self.measures:
                if measure in allowedMeasures:
                    measureMethod_ = getattr(importlib.import_module('deepclas4bio.'+'Measures'), measure)
                    if predictorName.endswith('DL4J'):
                        r = measureMethod_(predictions, self.labels,False)
                    else:
                        r = measureMethod_(predictions, self.labels)
                    resultMeasure[measure]=r
            result[predictorName]=resultMeasure
        return result