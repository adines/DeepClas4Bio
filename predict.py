import argparse
import json
from PredictorFactory import PredictorFactory

def predict(image,framework,model):
    predictor_factory=PredictorFactory()
    modelo=predictor_factory.getPredictor(framework,model)
    return modelo.predict(image)

parser=argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Image to classify")
parser.add_argument("-f", "--framework", required=True, help="Framework used to classify the image")
parser.add_argument("-m", "--model", required=True, help="Model used to classify the image")

args=vars(parser.parse_args())
result=predict(args["image"],args["framework"], args["model"])

data={}
data['type']='classification'
data['image']=args["image"]
data['framework']=args["framework"]
data['model']=args["model"]
data['class']=result
with open('data.json','w') as f:
    json.dump(data,f)