from deepclas4bio import Predictor

class FastaiPredictor(Predictor.Predictor):
    def predict(self,image):
        preProcessor=self.model.preProcessor
        imageProcessed=preProcessor(image)
        pred_class, pred_idx, outputs=self.model.deepModel.predict(imageProcessed)
        return pred_class

    def predictBatch(self,images):
        preProcessor = self.model.preProcessor
        output=[]
        for image in images:
            imageProcessed = preProcessor(image)
            pred_class, pred_idx, outputs = self.model.deepModel.predict(imageProcessed)
            output.append(outputs.numpy())
        return output