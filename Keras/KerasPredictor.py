import Predictor

class KerasPredictor(Predictor.Predictor):
    def predict(self,image):
        preProcessor=self.model.preProcessor
        imageProcessed=preProcessor(image)
        y_pred=self.model.deepModel.predict(imageProcessed)
        postProcessor=self.model.postProcessor
        return postProcessor(y_pred)