import sys
import json
from deepclas4bio.predict import predict as pred
from deepclas4bio.predict import predictBatch as predBatch
from deepclas4bio.listMethods import listFrameworks as lsFrameworks
from deepclas4bio.listMethods import listModels as lsModels
from deepclas4bio.listMethods import listMeasures as lsMeasures
from deepclas4bio.evaluate import evaluate as ev
from deepclas4bio.listMethods import listReadDatasets as lsDataset

def predict():

    image = sys.argv[1]
    framework=sys.argv[2]
    model=sys.argv[3]

    result = pred(image, framework, model)

    data = {}
    data['type'] = 'classification'
    data['image'] = image
    data['framework'] = framework
    data['model'] = model
    data['class'] = result
    with open('data.json', 'w') as f:
        json.dump(data, f)


def predictBatch():

    with open(sys.argv[1]) as file:
        data = json.load(file)

    framework = data['framework']
    model = data['model']
    images = data['images']

    results = predBatch(images, framework, model)

    resultImages = []
    for image, result in zip(images, results):
        object = {}
        object['image'] = image
        object['class'] = result
        resultImages.append(object)

    data = {}
    data['type'] = 'classification'
    data['framework'] = framework
    data['model'] = model
    data['results'] = resultImages
    with open('data.json', 'w') as f:
        json.dump(data, f)


def listFrameworks():
    data = {}
    data['type'] = 'frameworks'
    data['frameworks'] = lsFrameworks()
    with open('data.json', 'w') as f:
        json.dump(data, f)


def listModels():
    framework=sys.argv[1]
    data = {}
    data['type'] = 'models'
    data['framework'] = framework
    data['models'] = lsModels(framework)
    with open('data.json', 'w') as f:
        json.dump(data, f)

def listMeasures():
    data = {}
    data['type'] = 'measures'
    data['measures'] = lsMeasures()
    with open('data.json', 'w') as f:
        json.dump(data, f)

def evaluate():
    config=sys.argv[1]

    with open(config) as file:
        data=json.load(file)


    readDataset=data['readDataset']
    path = data['pathDataset']
    pathLabels = data['pathLabels']
    measures = data['measures']
    predictors = data['predictors']

    result=ev(readDataset,path,pathLabels,measures,predictors)
    with open('data.json', 'w') as f:
        json.dump(result, f)

def listReadDatasets():
    data = {}
    data['type'] = 'readDatasets'
    data['readDatasets'] = lsDataset()
    with open('data.json', 'w') as f:
        json.dump(data, f)

