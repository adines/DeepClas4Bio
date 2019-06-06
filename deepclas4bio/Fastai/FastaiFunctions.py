from fastai.vision import *
from pathlib import Path
import os
import requests

models=['ResNet34Kvasir']


######## METHODS FOR LOAD MODELS ########
def resnet34kvasirfastaiload():
    return loadModel('ResNet34Kvasir')


# Generic method to laod models from name
def loadModel(modelName):
    path = str(Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'Fastai' + os.sep
    pathModel = path + 'Classification' + os.sep + 'model'
    name=modelName+'.pkl'
    loaded_model=load_learner(pathModel,name)
    return loaded_model



######## METHODS FOR PREPROCESS ########
def commonPreProcess(im):
    return open_image(im)

def resnet34kvasirfastaipreprocess(im):
    return commonPreProcess(im)



######## METHODS FOR POSPROCESS ########
def commonPostProcess(result):
    path = str(Path.home()) + os.sep + 'DeepClas4BioModels'
    if not os.path.exists(path + os.sep + 'synset_words.txt'):
        if not os.path.exists(path):
            os.makedirs(path)
        r = requests.get(
            'https://unirioja-my.sharepoint.com/:t:/g/personal/adines_unirioja_es/ERS2ZWkLvc1AqY8FqIEjKBQB8MMobadwzrWsw4g86DBdAg?download=1')
        with open(path + os.sep + 'synset_words.txt', 'wb') as f:
            f.write(r.content)
    labels = np.loadtxt(path +os.sep+ "synset_words.txt", str, delimiter='\n')
    return labels[result.argmax()]


def resnet34kvasirfastaipostprocess(result):
    labels = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus',
              'normal-z-line', 'polyps', 'ulcerative-colitis']
    return labels[result.argmax()]
