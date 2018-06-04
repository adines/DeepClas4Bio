
class Model:

    def __init__(self,loadModel,preProcess,postProcess):
        self.deepModel=loadModel()
        self.preProcessor=preProcess
        self.postProcessor=postProcess