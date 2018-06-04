import inspect
import os
import numpy as np
import torch
from PIL import Image
import importlib
from torchvision import transforms
from torch.autograd import Variable
import torchvision.models as models


######## METHODS FOR LOAD MODELS ########

# Generic method to laod models from name
def loadModel(modelName):
    framework="PyTorch"
    class_name=modelName
    package_name=framework+".Classification.model"
    class__=getattr(importlib.import_module(package_name+"."+class_name),class_name)
    modelo=class__()


    path = inspect.stack()[0][1]
    pos = path.rfind(os.sep)
    path=path[:pos+1]


    pathWeights ='Classification' + os.sep + 'weights' + os.sep + modelName + '.pth'
    pathWeights=path+pathWeights
    modelo.load_state_dict(torch.load(pathWeights))
    return modelo


def vgg11pytorchload():
    return models.vgg11(pretrained=True)

def vgg13pytorchload():
    return models.vgg13(pretrained=True)

def vgg16pytorchload():
    return models.vgg16(pretrained=True)

def vgg19pytorchload():
    return models.vgg19(pretrained=True)

def alexnetpytorchload():
    return models.alexnet(pretrained=True)

def densenet121pytorchload():
    return models.densenet121(pretrained=True)

def densenet161pytorchload():
    return models.densenet161(pretrained=True)

def densenet169pytorchload():
    return models.densenet169(pretrained=True)

def densenet201pytorchload():
    return models.densenet201(pretrained=True)

def inceptionv3pytorchload():
    return models.inception_v3(pretrained=True)

def resnet18pytorchload():
    return models.resnet18(pretrained=True)

def resnet34pytorchload():
    return models.resnet34(pretrained=True)

def resnet50pytorchload():
    return models.resnet50(pretrained=True)

def resnet101pytorchload():
    return models.resnet101(pretrained=True)

def resnet152pytorchload():
    return models.resnet152(pretrained=True)

def squeezenet10pytorchload():
    return models.squeezenet1_0(pretrained=True)

def squeezenet11pytorchload():
    return models.squeezenet1_1(pretrained=True)


######## METHODS FOR PREPROCESS ########
def commonPreProcess(im):
    img_pil=Image.open(im)
    prep=transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    img_prep=prep(img_pil)
    img_var=Variable(img_prep)
    img_var=img_var.unsqueeze(0)
    return img_var


def vgg16pytorchpreprocess(im):
    return commonPreProcess(im)


def vgg19pytorchpreprocess(im):
    return commonPreProcess(im)


def caffenetpytorchpreprocess(im):
    return commonPreProcess(im)


def inceptionv3pytorchpreprocess(im):
    return commonPreProcess(im)


def ninpytorchpreprocess(im):
    return commonPreProcess(im)


def residualnet152pytorchpreprocess(im):
    return commonPreProcess(im)


def resnet101pytorchpreprocess(im):
    return commonPreProcess(im)


def squeezenetpytorchpreprocess(im):
    return commonPreProcess(im)


######## METHODS FOR POSPROCESS ########
def commonPostProcess(result):
    path = inspect.stack()[0][1]
    pos = path.rfind(os.sep)
    path = path[:pos + 1]
    labels = np.loadtxt(path + "synset_words.txt", str, delimiter='\n')
    return labels[result.data.numpy().argmax()]


def vgg11pytorchpostprocess(result):
    return commonPostProcess(result)

def vgg13pytorchpostprocess(result):
    return commonPostProcess(result)

def vgg16pytorchpostprocess(result):
    return commonPostProcess(result)

def vgg19pytorchpostprocess(result):
    return commonPostProcess(result)

def alexnetpytorchpostrocess(result):
    return commonPostProcess(result)

def densenet121pytorchpostrocess(result):
    return commonPostProcess(result)

def densenet161pytorchpostrocess(result):
    return commonPostProcess(result)

def densenet169pytorchpostrocess(result):
    return commonPostProcess(result)

def densenet201pytorchpostrocess(result):
    return commonPostProcess(result)

def inceptionv3pytorchpostprocess(result):
    return commonPostProcess(result)

def resnet18pytorchpostprocess(result):
    return commonPostProcess(result)

def resnet34pytorchpostprocess(result):
    return commonPostProcess(result)

def resnet50pytorchpostprocess(result):
    return commonPostProcess(result)

def resnet101pytorchpostprocess(result):
    return commonPostProcess(result)

def resnet152pytorchpostprocess(result):
    return commonPostProcess(result)

def squeezenet10pytorchpostprocess(result):
    return commonPostProcess(result)

def squeezenet11pytorchpostprocess(result):
    return commonPostProcess(result)