import inspect
import os
import numpy as np
import mxnet as mx
from PIL import Image

# Add your model here
models=['VGG16','VGG19','CaffeNet','InceptionV3','NiN', 'ResidualNet152', 'ResNet101', 'SqueezeNet']



######## METHODS FOR LOAD MODELS ########

# Generic method to laod models from name
def loadModel(modelName):
    path=inspect.stack()[0][1]
    pos=path.rfind(os.sep)
    pathModel=path[:pos+1]+'Classification'+os.sep+'model'+os.sep+modelName+'.json'
    symbol=mx.sym.load(pathModel)
    pathWeights=path[:pos+1]+'Classification'+os.sep+'weights'+os.sep+modelName+'.params'
    save_dict=mx.nd.load(pathWeights)
    arg_params={}
    aux_params={}
    for k, v in save_dict.items():
        tp, name=k.split(':',1)
        if tp=='arg':
            arg_params[name]=v
        elif tp=='aux':
            aux_params[name]=v
        mod=mx.mod.Module(symbol=symbol,context=mx.cpu())
        mod.bind(for_training=False,data_shapes=[('data', (1,3,224,224))])
        mod.set_params(arg_params,aux_params,allow_missing=True)
    return mod

def vgg16mxnetload():
    return loadModel('VGG16')

def vgg19mxnetload():
    return loadModel('VGG19')

def caffenetmxnetload():
    return loadModel('CaffeNet')

def inceptionv3mxnetload():
    return loadModel('InceptionV3')

def ninmxnetload():
    return loadModel('NiN')

def residualnet152mxnetload():
    return loadModel('ResidualNet152')

def resnet101mxnetload():
    return loadModel('ResNet101')

def squeezenetmxnetload():
    return loadModel('SqueezeNet')


######## METHODS FOR PREPROCESS ########
def commonPreProcess(im):
    img=Image.open(im)
    img=img.resize((224,224),Image.NEAREST)
    img_arr=np.array(img.getdata()).astype(np.float32).reshape((img.size[0],img.size[1],3))
    img_arr=np.swapaxes(img_arr,0,2)
    img_arr=np.swapaxes(img_arr,1,2)
    img_arr=img_arr[np.newaxis,:]
    return img_arr

def vgg16mxnetpreprocess(im):
    return commonPreProcess(im)

def vgg19mxnetpreprocess(im):
    return commonPreProcess(im)

def caffenetmxnetpreprocess(im):
    return commonPreProcess(im)

def inceptionv3mxnetpreprocess(im):
    return commonPreProcess(im)

def ninmxnetpreprocess(im):
    return commonPreProcess(im)

def residualnet152mxnetpreprocess(im):
    return commonPreProcess(im)

def resnet101mxnetpreprocess(im):
    return commonPreProcess(im)

def squeezenetmxnetpreprocess(im):
    return commonPreProcess(im)


######## METHODS FOR POSPROCESS ########
def commonPostProcess(result):
    result=np.squeeze(result)
    result=np.argsort(result)[::-1]
    path=inspect.stack()[0][1]
    pos=path.rfind(os.sep)
    path=path[:pos+1]
    labels=np.loadtxt(path+"synset_words.txt",str,delimiter='\n')
    return labels[result[0]]


def vgg16mxnetpostprocess(result):
    return commonPostProcess(result)

def vgg19mxnetpostprocess(result):
    return commonPostProcess(result)

def caffenetmxnetpostrocess(result):
    return commonPostProcess(result)

def inceptionv3mxnetpostprocess(result):
    return commonPostProcess(result)

def ninmxnetpostprocess(result):
    return commonPostProcess(result)

def residualnet152mxnetpostprocess(result):
    return commonPostProcess(result)

def resnet101mxnetpostprocess(result):
    return commonPostProcess(result)

def squeezenetmxnetpostprocess(result):
    return commonPostProcess(result)