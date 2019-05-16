import os
import numpy as np
import mxnet as mx
from mxnet import gluon,nd
from mxnet.gluon.model_zoo import vision
from pathlib import Path
import requests
import skimage.io as io

# Add your model here
models=['VGG11','VGG13','VGG16','VGG19','DenseNet121','DenseNet161','DenseNet169','DenseNet201','InceptionV3','AlexNet',
        'ResNet18','ResNet34','ResNet50','ResNet101','SqueezeNet','MobileNet','ResNet34Kvasir']



######## METHODS FOR LOAD MODELS ########

# Generic method to laod models from name
def loadModel(modelName):
    path = str(Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'MxNet' + os.sep
    pathModel=path+'Classification'+os.sep+'model'+os.sep+modelName+'.json'
    pathWeights=path+'Classification'+os.sep+'weights'+os.sep+modelName+'.params'


    ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    deserialized_net=gluon.nn.SymbolBlock.imports(pathModel, ['data'], pathWeights, ctx=ctx)

    return deserialized_net

def vgg11mxnetload():
    net=vision.vgg11(pretrained=True)
    net.hybridize()
    return net

def vgg13mxnetload():
    net=vision.vgg13(pretrained=True)
    net.hybridize()
    return net

def vgg16mxnetload():
    net= vision.vgg16(pretrained=True)
    net.hybridize()
    return net

def vgg19mxnetload():
    net=vision.vgg19(pretrained=True)
    net.hybridize()
    return net

def densenet121mxnetload():
    net = vision.densenet121(pretrained=True)
    net.hybridize()
    return net

def densenet161mxnetload():
    net = vision.densenet161(pretrained=True)
    net.hybridize()
    return net

def densenet169mxnetload():
    net = vision.densenet169(pretrained=True)
    net.hybridize()
    return net

def densenet201mxnetload():
    net = vision.densenet201(pretrained=True)
    net.hybridize()
    return net


def inceptionv3mxnetload():
    net = vision.inception_v3(pretrained=True)
    net.hybridize()
    return net

def alexnetmxnetload():
    net = vision.alexnet(pretrained=True)
    net.hybridize()
    return net

def resnet18mxnetload():
    net = vision.resnet18_v2(pretrained=True)
    net.hybridize()
    return net

def resnet34mxnetload():
    net = vision.resnet34_v2(pretrained=True)
    net.hybridize()
    return net

def resnet50mxnetload():
    net = vision.resnet50_v2(pretrained=True)
    net.hybridize()
    return net

def resnet101mxnetload():
    net = vision.resnet101_v2(pretrained=True)
    net.hybridize()
    return net

def resnet152mxnetload():
    net = vision.resnet152_v2(pretrained=True)
    net.hybridize()
    return net

def squeezenetmxnetload():
    net = vision.squeezenet1_1(pretrained=True)
    net.hybridize()
    return net


def mobilenetmxnetload():
    net = vision.mobilenet_v2_1_0(pretrained=True)
    net.hybridize()
    return net


def resnet34kvasirmxnetload():
    net=loadModel('ResNet34Kvasir')
    net.hybridize()
    return net



######## METHODS FOR PREPROCESS ########
def commonPreProcess(im):
    img=mx.nd.array(io.imread(im)).astype(np.uint8)
    img=mx.image.resize_short(img,256)
    img,_=mx.image.center_crop(img,(224,224))
    img=mx.image.color_normalize(img.astype(np.float32)/255,
                                 mean=mx.nd.array([0.485,0.456,0.406]),
                                 std=mx.nd.array([0.229,0.224,0.225]))
    img=mx.nd.transpose(img.astype('float32'),(2,1,0))
    return img

def vgg11mxnetpreprocess(im):
    return commonPreProcess(im)

def vgg13mxnetpreprocess(im):
    return commonPreProcess(im)

def vgg16mxnetpreprocess(im):
    return commonPreProcess(im)

def vgg19mxnetpreprocess(im):
    return commonPreProcess(im)

def densenet121mxnetpreprocess(im):
    return commonPreProcess(im)

def densenet161mxnetpreprocess(im):
    return commonPreProcess(im)

def densenet169mxnetpreprocess(im):
    return commonPreProcess(im)

def densenet201mxnetpreprocess(im):
    return commonPreProcess(im)

def inceptionv3mxnetpreprocess(im):
    return commonPreProcess(im)

def alexnetmxnetpreprocess(im):
    return commonPreProcess(im)

def resnet18mxnetpreprocess(im):
    return commonPreProcess(im)

def resnet34mxnetpreprocess(im):
    return commonPreProcess(im)

def resnet50mxnetpreprocess(im):
    return commonPreProcess(im)

def resnet101mxnetpreprocess(im):
    return commonPreProcess(im)

def resnet152mxnetpreprocess(im):
    return commonPreProcess(im)

def squeezenetmxnetpreprocess(im):
    return commonPreProcess(im)

def mobilenetmxnetpreprocess(im):
    return commonPreProcess(im)

def resnet34kvasirmxnetpreprocess(im):
    img = mx.nd.array(io.imread(im)).astype(np.uint8)
    img=mx.image.resize_short(img,227)
    img,_=mx.image.center_crop(img,(227,227))
    #     mg=mx.image.color_normalize(img.astype(np.float32)/255,
    #                                  mean=mx.nd.array([0.485,0.456,0.406]),
    #                                  std=mx.nd.array([0.229,0.224,0.225]))
    img = mx.nd.transpose(img.astype('float32'), (2, 1, 0))
    # img = mx.nd.expand_dims(img, axis=0)
    return img

######## METHODS FOR POSPROCESS ########
def commonPostProcess(result):
    result=np.squeeze(result)
    result=np.argsort(result)[::-1]
    path = str(Path.home()) + os.sep + 'DeepClas4BioModels'
    if not os.path.exists(path + os.sep + 'synset_words.txt'):
        if not os.path.exists(path):
            os.makedirs(path)
        r = requests.get(
            'https://unirioja-my.sharepoint.com/:t:/g/personal/adines_unirioja_es/ERS2ZWkLvc1AqY8FqIEjKBQB8MMobadwzrWsw4g86DBdAg?download=1')
        with open(path + os.sep + 'synset_words.txt', 'wb') as f:
            f.write(r.content)
    labels=np.loadtxt(path+os.sep+"synset_words.txt",str,delimiter='\n')
    return labels[result[0]]


def vgg11mxnetpostprocess(result):
    return commonPostProcess(result)

def vgg13mxnetpostprocess(result):
    return commonPostProcess(result)

def vgg16mxnetpostprocess(result):
    return commonPostProcess(result)

def vgg19mxnetpostprocess(result):
    return commonPostProcess(result)

def densenet121mxnetpostrocess(result):
    return commonPostProcess(result)

def densenet161mxnetpostrocess(result):
    return commonPostProcess(result)

def densenet169mxnetpostrocess(result):
    return commonPostProcess(result)

def densenet201mxnetpostrocess(result):
    return commonPostProcess(result)

def inceptionv3mxnetpostprocess(result):
    return commonPostProcess(result)

def alexnetmxnetpostprocess(result):
    return commonPostProcess(result)

def resnet18mxnetpostprocess(result):
    return commonPostProcess(result)

def resnet34mxnetpostprocess(result):
    return commonPostProcess(result)

def resnet50mxnetpostprocess(result):
    return commonPostProcess(result)

def resnet101mxnetpostprocess(result):
    return commonPostProcess(result)

def resnet152mxnetpostprocess(result):
    return commonPostProcess(result)

def squeezenetmxnetpostprocess(result):
    return commonPostProcess(result)

def mobilenetmxnetpostprocess(result):
    return commonPostProcess(result)

def resnet34kvasirmxnetpostprocess(result):
    result=np.squeeze(result)
    # result=result.asnumpy()
    result=np.ndarray.argsort(result)[::-1]
    labels=['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
    return labels[result[0]]