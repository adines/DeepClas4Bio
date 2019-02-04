import numpy as np
import inspect
import os
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
    path = inspect.stack()[0][1]
    pos = path.rfind(os.sep)
    path = path[:pos + 1]
    labels = np.loadtxt(path + "synset_words.txt", str, delimiter='\n')
    return labels[result[0].argmax()]

def vgg16dl4jpostprocess(im):
    return commonPostProcess(im)

def vgg19dl4jpostprocess(im):
    return commonPostProcess(im)

def resnet50dl4jpostprocess(im):
    return commonPostProcess(im)

def googlenetdl4jpostprocess(im):
    return commonPostProcess(im)
