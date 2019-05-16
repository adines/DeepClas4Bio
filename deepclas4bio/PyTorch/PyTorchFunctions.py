import os
import numpy as np
import torch
from PIL import Image
import importlib
from torchvision import transforms
from torch.autograd import Variable
import torchvision.models as torchmodels
from pathlib import Path
import sys
import requests

# Add your model here
models=['VGG11','VGG13','VGG16','VGG19','AlexNet','DenseNet121','DenseNet161','DenseNet169','DenseNet201','InceptionV3',
        'ResNet18','ResNet34','ResNet50','ResNet101','ResNet152','SqueezeNet10','SqueezeNet11','ResNet34Kvasir']



######## METHODS FOR LOAD MODELS ########

# Generic method to laod models from name
def loadModel(modelName):
    sys.path.append(os.path.abspath(str(Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'PyTorch' + os.sep+'Classification' + os.sep + 'model'))
    class_name=modelName
    class__=getattr(importlib.import_module(class_name),class_name)
    modelo=class__()

    path = str(Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'PyTorch' + os.sep


    pathWeights ='Classification' + os.sep + 'weights' + os.sep + modelName + '.pth'
    pathWeights=path+pathWeights
    state=torch.load(pathWeights,torch.device('cpu'))
    modelo.load_state_dict(state_dict=state, strict=False)
    modelo=modelo.eval()
    return modelo


def vgg11pytorchload():
    model= torchmodels.vgg11(pretrained=True)
    return model.eval()

def vgg13pytorchload():
    model= torchmodels.vgg13(pretrained=True)
    return model.eval()

def vgg16pytorchload():
    model= torchmodels.vgg16(pretrained=True)
    return model.eval()

def vgg19pytorchload():
    return torchmodels.vgg19(pretrained=True)

def alexnetpytorchload():
    model= torchmodels.alexnet(pretrained=True)
    return model.eval()

def densenet121pytorchload():
    model= torchmodels.densenet121(pretrained=True)
    return model.eval()

def densenet161pytorchload():
    model= torchmodels.densenet161(pretrained=True)
    return model.eval()

def densenet169pytorchload():
    model= torchmodels.densenet169(pretrained=True)
    return model.eval()

def densenet201pytorchload():
    model= torchmodels.densenet201(pretrained=True)
    return model.eval()

def inceptionv3pytorchload():
    model= torchmodels.inception_v3(pretrained=True)
    return model.eval()

def resnet18pytorchload():
    model= torchmodels.resnet18(pretrained=True)
    return model.eval()

def resnet34pytorchload():
    model= torchmodels.resnet34(pretrained=True)
    return model.eval()

def resnet50pytorchload():
    model= torchmodels.resnet50(pretrained=True)
    return model.eval()

def resnet101pytorchload():
    model= torchmodels.resnet101(pretrained=True)
    return model.eval()

def resnet152pytorchload():
    model= torchmodels.resnet152(pretrained=True)
    return model.eval()

def squeezenet10pytorchload():
    model= torchmodels.squeezenet1_0(pretrained=True)
    return model.eval()

def squeezenet11pytorchload():
    model= torchmodels.squeezenet1_1(pretrained=True)
    return model.eval()

def resnet34kvasirpytorchload():
    pathModel = str(
        Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'PyTorch' + os.sep + 'Classification' + os.sep + 'model'
    pathWeights = str(
        Path.home()) + os.sep + 'DeepClas4BioModels' + os.sep + 'PyTorch' + os.sep + 'Classification' + os.sep + 'weights'
    if not os.path.exists(pathModel + os.sep + 'ResNet34Kvasir.py'):
        if not os.path.exists(pathModel):
            os.makedirs(pathModel)

        r = requests.get('https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/Edg-3LK7P5RBggEfkOU86fIByryGl3tmsvqhuBoGsXZ91A?download=1')
        with open(pathModel + os.sep + 'ResNet34Kvasir.py', 'wb') as f:
            f.write(r.content)
    if not os.path.exists(pathWeights + os.sep + 'ResNet34Kvasir.pth'):
        if not os.path.exists(pathWeights):
            os.makedirs(pathWeights)
        r=requests.get(
            'https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EVlYtl3fsjVLp2SjhGrZAbYB3HtYxa9PWM2y9eHI4PKIlw?download=1')
        with open(pathWeights + os.sep + 'ResNet34Kvasir.pth', 'wb') as f:
            f.write(r.content)

    return loadModel('ResNet34Kvasir')


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


def vgg11pytorchpreprocess(im):
    return commonPreProcess(im)

def vgg13pytorchpreprocess(im):
    return commonPreProcess(im)

def vgg16pytorchpreprocess(im):
    return commonPreProcess(im)


def vgg19pytorchpreprocess(im):
    return commonPreProcess(im)


def alexnetpytorchpreprocess(im):
    return commonPreProcess(im)

def densenet121pytorchpreprocess(im):
    return commonPreProcess(im)

def densenet161pytorchpreprocess(im):
    return commonPreProcess(im)

def densenet169pytorchpreprocess(im):
    return commonPreProcess(im)

def densenet201pytorchpreprocess(im):
    return commonPreProcess(im)

def inceptionv3pytorchpreprocess(im):
    return commonPreProcess(im)

def resnet18pytorchpreprocess(im):
    return commonPreProcess(im)

def resnet34pytorchpreprocess(im):
    return commonPreProcess(im)

def resnet50pytorchpreprocess(im):
    return commonPreProcess(im)

def resnet101pytorchpreprocess(im):
    return commonPreProcess(im)

def resnet152pytorchpreprocess(im):
    return commonPreProcess(im)

def squeezenet10pytorchpreprocess(im):
    return commonPreProcess(im)

def squeezenet11pytorchpreprocess(im):
    return commonPreProcess(im)

def resnet34kvasirpytorchpreprocess(im):
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
    labels = np.loadtxt(path + os.sep+"synset_words.txt", str, delimiter='\n')
    return labels[result.argmax()]


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

def resnet34kvasirpytorchpostprocess(result):
    labels = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus',
              'normal-z-line', 'polyps', 'ulcerative-colitis']
    return labels[result.argmax()]
