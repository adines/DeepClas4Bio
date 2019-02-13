from deepclas4bio import Measures
import importlib
import inspect
import os

def listFrameworks():
    path=inspect.stack()[0][1]
    pos=path.rfind(os.sep)
    path=path[:pos+1]
    frameworks=[]
    if(pos==-1):
        path='.'
    dirs=os.listdir(path=path)

    for d in dirs:
        if "." not in d and not d.startswith("__") and not d.startswith("temp"):
            frameworks.append(d)
    return frameworks


def listModels(framework):
    class_name = framework + 'Functions'
    models = getattr(importlib.import_module('deepclas4bio.'+framework + '.' + class_name), 'models')
    return models


def listMeasures():
    bm= Measures.binaryMeasures
    nbm= Measures.noBinaryMeasures
    return list(set().union(bm,nbm))

readDatasets=[{'name':'ReadDatasetFolders','description':'The images have to be organized in folders. Each folder must have the name of the class which the images belong.'}]
def listReadDatasets():
    return readDatasets