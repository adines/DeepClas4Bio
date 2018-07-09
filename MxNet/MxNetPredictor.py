import Predictor
import mxnet as mx

class MxNetPredictor(Predictor.Predictor):
    def predict(self,image):
        preProcessor=self.model.preProcessor
        imageProcessed=preProcessor(image)

        x=mx.io.NDArrayIter(imageProcessed)

        deepModel=self.model.deepModel
        y_preds=deepModel.predict(x)
        postProcessor=self.model.postProcessor
        return postProcessor(y_preds[0].asnumpy())