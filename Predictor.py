import IPredictor

class Predictor(IPredictor.IPredictor):

    def __init__(self,model):
        self.model=model

    def predict(self,image):
        pass