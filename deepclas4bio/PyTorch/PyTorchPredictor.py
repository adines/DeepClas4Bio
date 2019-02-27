from deepclas4bio import Predictor


class PyTorchPredictor(Predictor.Predictor):
    def predict(self,image):
        preProcessor=self.model.preProcessor
        imageProcessed=preProcessor(image)
        deepModel=self.model.deepModel
        y_preds=deepModel(imageProcessed)
        y_preds=y_preds.data.numpy()
        postProcessor=self.model.postProcessor
        return postProcessor(y_preds)

    def predictBatch(self,images):
        preProcessor = self.model.preProcessor
        deepModel = self.model.deepModel
        predictions = []
        for image in images:
            imageProcessed = preProcessor(image)
            y_preds = deepModel(imageProcessed)
            y_preds=y_preds.data.numpy()
            predictions.append(y_preds)
        return predictions
