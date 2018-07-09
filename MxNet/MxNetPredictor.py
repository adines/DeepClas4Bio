import Predictor
import mxnet as mx
import numpy as np

class MxNetPredictor(Predictor.Predictor):
    def predict(self,image):
        preProcessor=self.model.preProcessor
        imageProcessed=preProcessor(image)

        x=mx.io.NDArrayIter(imageProcessed)

        deepModel=self.model.deepModel
        y_preds=deepModel.predict(x)
        postProcessor=self.model.postProcessor
        return postProcessor(y_preds[0].asnumpy())

    def predictBatch(self,images):
        preProcessor = self.model.preProcessor
        data=[]
        for image in images:
            imageProcessed = preProcessor(image)
            data.append(imageProcessed)
        dataStack=np.vstack(data)
        dataMx = mx.io.NDArrayIter(dataStack)
        y_pred = self.model.deepModel.predict(dataMx)
        predictions=[]
        for pred in y_pred:
            x = pred.asnumpy()
            x = np.squeeze(x)
            x=np.argsort(x)[::-1]
            predictions.append(x)
        return predictions