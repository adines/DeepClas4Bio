import argparse
import json
import DatasetManager
from PredictorFactory import PredictorFactory

def predictBatch(images,framework,model,batch=64):
    predictor_factory=PredictorFactory()
    modelo=predictor_factory.getPredictor(framework,model)
    dataManager=DatasetManager.DatasetManager(images, batch=batch)
    predictions=[]
    while(dataManager.hasNextBach()):
        batchImages=dataManager.nextBatch()
        prediction=modelo.predictBatch(batchImages)
        predictions+=prediction
    results=[]
    for p in predictions:
        results.append(modelo.model.postProcessor(p))
    return results

parser=argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True, help="Path to the config.json file")

args=vars(parser.parse_args())
with open(args["config"]) as file:
        data=json.load(file)

framework=data['framework']
model=data['model']
images=data['images']

results=predictBatch(images,framework,model)

resultImages=[]
for image, result in zip(images,results):
    object={}
    object['image']=image
    object['class']=result
    resultImages.append(object)

data={}
data['type']='classification'
data['framework']=framework
data['model']=model
data['results']=resultImages
with open('data.json','w') as f:
    json.dump(data,f)
