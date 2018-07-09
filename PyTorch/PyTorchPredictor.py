import Predictor
import numpy as np

class PyTorchPredictor(Predictor.Predictor):
    def predict(self,image):
        preProcessor=self.model.preProcessor
        imageProcessed=preProcessor(image)
        deepModel=self.model.deepModel
        y_preds=deepModel(imageProcessed)
        postProcessor=self.model.postProcessor
        return postProcessor(y_preds)

    def predictBatch(self,images):
        preProcessor = self.model.preProcessor
        deepModel = self.model.deepModel
        predictions = []
        for image in images:
            imageProcessed = preProcessor(image)
            y_preds = deepModel(imageProcessed)
            x=y_preds.data.numpy()
            x=np.argsort(x)[::-1]
            predictions.append(x)
        return predictions