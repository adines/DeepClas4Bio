import Predictor
from collections import namedtuple
import mxnet as mx

class MxNetPredictor(Predictor.Predictor):
    def predict(self,image):
        preProcessor=self.model.preProcessor
        imageProcessed=preProcessor(image)
        Batch=namedtuple('Batch',['data'])
        deepModel=self.model.deepModel
        deepModel.forward(Batch([mx.nd.array(imageProcessed)]))
        postProcessor=self.model.postProcessor
        return postProcessor(deepModel.get_outputs()[0].asnumpy())