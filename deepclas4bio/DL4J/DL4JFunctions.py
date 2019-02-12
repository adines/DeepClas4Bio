import numpy as np
import os
import requests
from pathlib import Path

# Add your model here
models=['VGG16','VGG19','ResNet50','GoogLeNet']

def vgg16dl4jpreprocess(im):
    pass

def vgg19dl4jpreprocess(im):
    pass

def resnet50dl4jpreprocess(im):
    pass

def googlenetdl4jpreprocess(im):
    pass



def vgg16dl4jload():
    pass

def vgg19dl4jload():
    pass

def resnet50dl4jload():
    pass

def googlenetdl4jload():
    pass


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
    result=np.asarray(result)
    return labels[result.argmax()]

def vgg16dl4jpostprocess(result):
    return commonPostProcess(result)

def vgg19dl4jpostprocess(result):
    return commonPostProcess(result)

def resnet50dl4jpostprocess(result):
    return commonPostProcess(result)

def googlenetdl4jpostprocess(result):
    return commonPostProcess(result)
