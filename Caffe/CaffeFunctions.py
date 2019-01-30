import inspect
import os
import numpy as np
#import sys
#sys.path.append(r'C:\Users\adines\Downloads\caffe\python')
import caffe

# Add your model here
models=['VGG16','VGG19','AlexNet','CaffeNet','GoogleNet']

######## METHODS FOR LOAD MODELS ########

# Generic method to laod models from name
def loadModel(modelName):
    path=inspect.stack()[0][1]
    pos=path.rfind(os.sep)
    path=path[:pos+1]
    pathModel='Classification'+os.sep+'model'+os.sep+modelName+'.prototxt'
    pathModel=path+pathModel
    pathWeights='Classification'+os.sep+'weights'+os.sep+modelName+'.caffemodel'
    pathWeights=path+pathWeights
    net=caffe.Classifier(pathModel,pathWeights,
                         mean=np.load(path+'ilsvrc_2012_mean.npy').mean(1).mean(1),
                         channel_swap=(2,1,0),
                         raw_scale=225,
                         image_dims=(255,255))
    return net


def vgg16caffeload():
    return loadModel('VGG16')

def vgg19caffeload():
    return loadModel('VGG19')

def alexnetcaffeload():
    return loadModel('AlexNet')

def caffenetcaffeload():
    return loadModel('CaffeNet')

def googlenetcaffeload():
    return loadModel('GoogleNet')


######## METHODS FOR PREPROCESS ########
def commonPreProcess(im):
    input_image=caffe.io.load_image(im)
    return input_image

def vgg16caffepreprocess(im):
    return commonPreProcess(im)

def vgg19caffepreprocess(im):
    return commonPreProcess(im)

def alexnetcaffepreprocess(im):
    return commonPreProcess(im)

def caffenetcaffepreprocess(im):
    return commonPreProcess(im)

def googlenetcaffepreprocess(im):
    return commonPreProcess(im)


######## METHODS FOR POSPROCESS ########
def commonPostProcess(result):
    path = inspect.stack()[0][1]
    pos = path.rfind(os.sep)
    path = path[:pos + 1]
    labels = np.loadtxt(path + "synset_words.txt", str, delimiter='\n')
    return labels[result[0].argmax()]

def vgg16caffepostprocess(result):
    return commonPostProcess(result)

def vgg19caffepostprocess(result):
    return commonPostProcess(result)

def alexnetcaffepostrocess(result):
    return commonPostProcess(result)

def caffenetcaffepostprocess(result):
    return commonPostProcess(result)

def googlenetcaffepostprocess(result):
    return commonPostProcess(result)