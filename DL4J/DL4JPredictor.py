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
        return result

    def predict(self,image):
        path=inspect.stack()[0][1]
        pos=path.rfind(os.sep)
        path=path[:pos+1]+'PredictDL4J.jar'

        args=[path,self.model,image]
        result=self.jarWrapper(*args)
        return result[0]

    def predictBatch(self,images):
        path = inspect.stack()[0][1]
        pos = path.rfind(os.sep)
        path = path[:pos + 1] + 'PredictDL4J.jar'

        args = [path, self.model]+images
        result = self.jarWrapper(*args)
        newResult=[]
        for r in result:
            r=r.replace("[","").replace("]","")
            newResult.append(list(map(int,map(float,r.split(",")))))
        return newResult