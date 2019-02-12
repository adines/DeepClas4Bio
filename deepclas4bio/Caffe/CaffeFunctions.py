import os
import numpy as np
#import sys
#sys.path.append(r'C:\Users\adines\Downloads\caffe\python')
import caffe
from pathlib import Path
import requests

# Add your model here
models=['VGG16','VGG19','AlexNet','CaffeNet','GoogleNet']

######## METHODS FOR LOAD MODELS ########

# Generic method to laod models from name
def loadModel(modelName):

    path = str(Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'Caffe' + os.sep

    pathModel='Classification'+os.sep+'model'+os.sep+modelName+'.prototxt'
    pathModel=path+pathModel
    pathWeights='Classification'+os.sep+'weights'+os.sep+modelName+'.caffemodel'
    pathWeights=path+pathWeights
    net=caffe.Classifier(pathModel,pathWeights,
                         channel_swap=(2,1,0),
                         raw_scale=225,
                         image_dims=(255,255))
    return net

def downloadModel(modelName,urlModel,urlWeights):
    pathModel = str(Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'Caffe' + os.sep + 'Classification' + os.sep + 'model'
    pathWeights = str(Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'Caffe' + os.sep + 'Classification' + os.sep + 'weights'
    if not os.path.exists(pathModel + os.sep + modelName+'.prototxt'):
        if not os.path.exists(pathModel):
            os.makedirs(pathModel)
        r = requests.get(urlModel)
        with open(pathModel + os.sep + modelName+'.prototxt', 'wb') as f:
            f.write(r.content)
    if not os.path.exists(pathWeights + os.sep + modelName+'.caffemodel'):
        if not os.path.exists(pathWeights):
            os.makedirs(pathWeights)
        r = requests.get(urlWeights)
        with open(pathWeights + os.sep + modelName+'.caffemodel', 'wb') as f:
            f.write(r.content)


def vgg16caffeload():
    downloadModel('VGG16','https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EUbdJT9GgVJDsMVVrwk1vewBP-IpOfKwnJh3sSNeXdoX7g?download=1',
                  'https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EbA3miHHnbxFrPMPB8b0QkUBFxK2r4yvq2Hk4OcU1c9ryg?download=1')
    return loadModel('VGG16')

def vgg19caffeload():
    downloadModel('VGG19','https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EVRmVQK2uPdEk4E-xOOs88wBeLns11VEtT6QsKLztc7i7A?download=1',
                  'https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EVUQxa-7Ya1DklQgSgKDgt8B5RqAazeyHsIKuAzhPKDGTw?download=1')
    return loadModel('VGG19')

def alexnetcaffeload():
    downloadModel('AlexNet','https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/ERYGs4x82xRNgQr4PqLUXVoB336TjhpF1BPUjj05LMp6IQ?download=1',
                  'https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EQ_ZexbsIHtDpK5LiW0S_UoBhSgcUC7ttQuPu_DjqV39Pg?download=1')
    return loadModel('AlexNet')

def caffenetcaffeload():
    downloadModel('CaffeNet','https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/Ea8iImrYGAVNpBI1wYwhVisBYX_fWQNtdbLc228Em6Npug?download=1',
                  'https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EVt8e-ngX1FHv07nHRLXaAcB9dnwX9i33dP8D64df150DQ?download=1')
    return loadModel('CaffeNet')

def googlenetcaffeload():
    downloadModel('GoogleNet','https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EaNUNfz20hRLtGcWikgrRuQBwyyMyySc5sc-LVmz_TZypQ?download=1',
                  'https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EXGYXCCwjllPiN0DPFZJ5S8BewR7j6MrPwjuc206ONtiug?download=1')
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
    path = str(Path.home()) + os.sep + 'DeepClas4BioModels'
    if not os.path.exists(path + os.sep + 'synset_words.txt'):
        if not os.path.exists(path):
            os.makedirs(path)
        r = requests.get(
            'https://unirioja-my.sharepoint.com/:t:/g/personal/adines_unirioja_es/ERS2ZWkLvc1AqY8FqIEjKBQB8MMobadwzrWsw4g86DBdAg?download=1')
        with open(path + os.sep + 'synset_words.txt', 'wb') as f:
            f.write(r.content)
    labels = np.loadtxt(path +os.sep+ "synset_words.txt", str, delimiter='\n')
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