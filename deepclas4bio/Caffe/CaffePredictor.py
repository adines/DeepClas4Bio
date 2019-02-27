from deepclas4bio import Predictor
import numpy as np

class CaffePredictor(Predictor.Predictor):
    def predict(self,image):
        preProcessor=self.model.preProcessor
        imageProcessed=preProcessor(image)
        y_pred=self.model.deepModel.predict([imageProcessed])
        postProcessor=self.model.postProcessor
        return postProcessor(y_pred)

    def predictBatch(self,images):
        preProcessor = self.model.preProcessor
        data=[]
        for image in images:
            imageProcessed = preProcessor(image)
            data.append(imageProcessed)
        y_pred = self.model.deepModel.predict(data)
        predictions=[]
        for pred in y_pred:
            predictions.append(pred)
        return predictions
