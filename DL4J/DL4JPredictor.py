import Predictor
from subprocess import *
import inspect
import os

class DL4JPredictor(Predictor.Predictor):

    def jarWrapper(self,*args):
        process=Popen(['java', '-jar']+list(args),stdout=PIPE,stderr=PIPE)
        result=[]

        for line in process.stdout:
            result.append(line.decode('utf-8').rstrip('\n').rstrip('\n'))
        return result[0]

    def predict(self,image):
        path=inspect.stack()[0][1]
        pos=path.rfind(os.sep)
        path=path[:pos+1]+'PredictDL4J.jar'

        args=[path,image,self.model]
        result=self.jarWrapper(*args)
        return result