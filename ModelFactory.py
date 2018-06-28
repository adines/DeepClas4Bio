import importlib

class ModelFactory:
    loadedModels={}
    def getModel(self,framework,model):
        modelName=framework+model
        m=self.loadedModels.get(modelName)
        if m is not None:
            return m
        else:
            if framework=='DL4J':
                return model
            else:
                class_name=framework+'Model'
                class__=getattr(importlib.import_module(framework+'.'+class_name),class_name)
                loadModel=model+framework+'load'
                loadModel=loadModel.lower()
                loadModelMethod = getattr(importlib.import_module(framework + '.' + framework + 'Functions'), loadModel)
                preprocess=model+framework+'preprocess'
                preprocess=preprocess.lower()
                preprocessMethod=getattr(importlib.import_module(framework+'.'+framework+'Functions'),preprocess)
                postprocess = model + framework + 'postprocess'
                postprocess = postprocess.lower()
                postprocessMethod = getattr(importlib.import_module(framework + '.' + framework + 'Functions'), postprocess)
                m= class__(loadModelMethod,preprocessMethod,postprocessMethod)
                self.loadedModels[modelName]=m
                return m