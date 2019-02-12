from deepclas4bio import IPredictor


class Predictor(IPredictor.IPredictor):

    def __init__(self,model):
        self.model=model

    def predict(self,image):
        pass

    def predictBatch(self,images):
        pass