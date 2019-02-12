from deepclas4bio import Predictor
import mxnet as mx
import numpy as np

class MxNetPredictor(Predictor.Predictor):
    def predict(self,image):
        preProcessor=self.model.preProcessor
        imageProcessed=preProcessor(image)
        deepModel=self.model.deepModel
        imageProcessed=mx.ndarray.expand_dims(imageProcessed, axis=0)
        y_preds=deepModel(imageProcessed)
        postProcessor=self.model.postProcessor
        return postProcessor(y_preds[0].asnumpy())

    def predictBatch(self,images):
        preProcessor = self.model.preProcessor
        data=[]
        for image in images:
            imageProcessed = preProcessor(image)
            data.append(imageProcessed)
        dataStack=mx.nd.stack(*data,axis=0)
        y_pred = self.model.deepModel(dataStack)
        predictions=[]
        for pred in y_pred:
            x = pred.asnumpy()
            x = np.squeeze(x)
            x=np.argsort(x)[::-1]
            predictions.append(x)
        return predictions
