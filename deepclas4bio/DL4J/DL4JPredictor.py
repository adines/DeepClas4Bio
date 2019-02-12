from deepclas4bio import Predictor
from subprocess import *
import os
from pathlib import Path
import requests

class DL4JPredictor(Predictor.Predictor):

    def jarWrapper(self,*args):
        process=Popen(['java', '-jar']+list(args),stdout=PIPE,stderr=PIPE)
        result=[]

        for line in process.stdout:
            result.append(line.decode('utf-8').rstrip('\n').rstrip('\n'))
        return result

    def downloadJar(self):
        path = str(Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'DL4J'
        if not os.path.exists(path + os.sep + 'PredictDL4J.jar'):
            if not os.path.exists(path):
                os.makedirs(path)
            r = requests.get(
                'https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EYHxT1EyIeRJluozEP2YlbIBSd7TIVAyNbfCGPm-gqySyg?download=1')
            with open(path + os.sep + 'PredictDL4J.jar', 'wb') as f:
                f.write(r.content)


    def predict(self,image):
        self.downloadJar()
        path=str(Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'DL4J'
        path=path+os.sep+'PredictDL4J.jar'

        args=[path,self.model.name[:-4],image]
        result=self.jarWrapper(*args)
        return result[0]

    def predictBatch(self,images):
        self.downloadJar()
        path = str(Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'DL4J'
        path = path + os.sep + 'PredictDL4J.jar'

        args = [path, self.model.name[:-4]]+images
        result = self.jarWrapper(*args)
        newResult=[]
        for r in result:
            r=r.replace("[","").replace("]","")
            newResult.append(list(map(int,map(float,r.split(",")))))
        return newResult
