from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications import *
from keras.applications.vgg16 import preprocess_input as vgg16preprocess
from keras.applications.vgg19 import preprocess_input as vgg19preprocess
from keras.applications.resnet50 import preprocess_input as resnet50preprocess
from keras.applications.inception_v3 import preprocess_input as inceptionv3preprocess
from keras.applications.mobilenet import preprocess_input as mobileNetpreprocess
from keras.applications.xception import preprocess_input as xceptionpreprocess
from keras.applications.densenet import preprocess_input as densenetpreprocess
from keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnetv2preprocess
from keras.applications.nasnet import preprocess_input as nasnetpreprocess

from keras.applications.vgg16 import decode_predictions as vgg16decode
from keras.applications.vgg19 import decode_predictions as vgg19decode
from keras.applications.resnet50 import decode_predictions as resnet50decode
from keras.applications.inception_v3 import decode_predictions as inceptionv3decode
from keras.applications.mobilenet import decode_predictions as mobileNetdecode
from keras.applications.xception import decode_predictions as xceptiondecode
from keras.applications.densenet import decode_predictions as densenetdecode
from keras.applications.inception_resnet_v2 import decode_predictions as inceptionresnetv2decode
from keras.applications.nasnet import decode_predictions as nasnetdecode

from keras.applications.vgg19 import *
from keras.applications.resnet50 import *
from keras.applications.inception_v3 import *
from keras.applications.densenet import *
from keras.applications.inception_resnet_v2 import *
from keras.applications.nasnet import *
from keras.applications.mobilenet import *

import os
import numpy as np
from pathlib import Path
import requests

# Add your model here
models=['VGG16','VGG19','ResNet','InceptionV3','MobileNet','Xception','InceptionResNetV2', 'DenseNet', 'NASNet','ResNetISIC']


######## METHODS FOR LOAD MODELS ########
def vgg16kerasload():
    return VGG16()

def vgg19kerasload():
    return VGG19()

def resnet50kerasload():
    return ResNet50()

def inceptionv3kerasload():
    return InceptionV3()

def mobilenetkerasload():
    return MobileNet()

def xceptionkerasload():
    return Xception()

def densenetkerasload():
    return DenseNet()

def inceptionresnetv2kerasload():
    return InceptionResNetV2()

def nasnetkerasload():
    return NASNet()

# Generic method to laod models from name
def loadModel(modelName):
    path=str(Path.home())+os.sep+'DeepClas4BioModels'+os.sep+'Keras'+os.sep
    pathModel=path+'Classification'+os.sep+'model'+os.sep+modelName+'.json'
    json_file=open(pathModel,'r')
    loaded_model_json=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_model_json)
    pathWeights=path+'Classification'+os.sep+'weights'+os.sep+modelName+'.h5'
    loaded_model.load_weights(pathWeights)
    return loaded_model

def resnetisickerasload():
    pathModel=str(Path.home()) + os.sep + 'DeepClas4BioModels'+os.sep+'Keras'+os.sep+'Classification'+os.sep+'model'
    pathWeights=str(Path.home()) + os.sep + 'DeepClas4BioModels'+os.sep+'Keras'+os.sep+'Classification'+os.sep+'weights'
    if not os.path.exists(pathModel+os.sep+'ResNetISIC.json'):
        if not os.path.exists(pathModel):
            os.makedirs(pathModel)
        r = requests.get(
            'https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EVNjE2vxXAFJvBcMzke1iHYBVL0GGTCkX3WKCrVMzCLrdA?download=1')
        with open(pathModel+os.sep+'ResNetISIC.json', 'wb') as f:
            f.write(r.content)
    if not os.path.exists(pathWeights+ os.sep + 'ResNetISIC.h5'):
        if not os.path.exists(pathWeights):
            os.makedirs(pathWeights)
        r=requests.get('https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/Eb61Ld1zqdFJlMxANXS1ANABjRMZ-FZnLuadeGn-2bFnCA?download=1')
        with open(pathWeights+os.sep+'ResNetISIC.h5', 'wb') as f:
            f.write(r.content)
    return loadModel('ResNetISIC')


######## METHODS FOR PREPROCESS ########
def commonPreProcess(im):
    img=image.load_img(im,target_size=(224,224))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    return x

def vgg16keraspreprocess(im):
    x=commonPreProcess(im)
    return vgg16preprocess(x)

def vgg19keraspreprocess(im):
    x=commonPreProcess(im)
    return vgg19preprocess(x)

def renet50keraspreprocess(im):
    x=commonPreProcess(im)
    return resnet50preprocess(x)

def inceptionv3keraspreprocess(im):
    x=commonPreProcess(im)
    return inceptionv3preprocess(x)

def mobilenetkeraspreprocess(im):
    x=commonPreProcess(im)
    return mobileNetpreprocess(x)

def xceptionkeraspreprocess(im):
    x=commonPreProcess(im)
    return xceptionpreprocess(x)

def densenetkeraspreprocess(im):
    x=commonPreProcess(im)
    return densenetpreprocess(x)

def inceptionresnetv2keraspreprocess(im):
    x=commonPreProcess(im)
    return inceptionresnetv2preprocess(x)

def nasnetkeraspreprocess(im):
    x=commonPreProcess(im)
    return nasnetpreprocess(x)

def resnetisickeraspreprocess(im):
    return commonPreProcess(im)


######## METHODS FOR POSPROCESS ########
def vgg16keraspostprocess(result):
    prediction=vgg16decode(result,top=1)[0]
    return prediction[0][1]

def vgg19keraspostprocess(result):
    prediction=vgg19decode(result,top=1)[0]
    return prediction[0][1]

def resnet50keraspostrocess(result):
    prediction=resnet50decode(result,top=1)[0]
    return prediction[0][1]

def inceptionv3keraspostprocess(result):
    prediction=inceptionv3decode(result,top=1)[0]
    return prediction[0][1]

def mobilenetkeraspostprocess(result):
    prediction=mobileNetdecode(result,top=1)[0]
    return prediction[0][1]

def xceptionkeraspostprocess(result):
    prediction=xceptiondecode(result,top=1)[0]
    return prediction[0][1]

def densenetkeraspostprocess(result):
    prediction=densenetdecode(result,top=1)[0]
    return prediction[0][1]

def inceptionresnetv2keraspostprocess(result):
    prediction=inceptionresnetv2decode(result,top=1)[0]
    return prediction[0][1]

def nasnetkeraspostprocess(result):
    prediction=nasnetdecode(result,top=1)[0]
    return prediction[0][1]

def resnetisickeraspostprocess(result):
    max = np.argmax(result)
    labels = ["MSK-1", "MSK-2", "MSK-3", "MSK-4", "MSK-5", "SONIC", "UDA-1"]
    return labels[max]
