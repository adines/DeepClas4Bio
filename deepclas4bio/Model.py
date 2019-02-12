
class Model:

    def __init__(self,name,loadModel,preProcess,postProcess):
        self.deepModel=loadModel()
        self.preProcessor=preProcess
        self.postProcessor=postProcess
        self.name=name