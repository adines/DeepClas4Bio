import json
import argparse
import importlib


def listModels(framework):
    class_name = framework + 'Functions'
    models = getattr(importlib.import_module(framework + '.' + class_name), 'models')
    return models


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-f", "--framework",required=True, help="Framework to obtain its models")
    args=vars(parser.parse_args())

    data={}
    data['type']='models'
    data['framework']=args["framework"]
    data['models']=listModels(args["framework"])
    with open('data.json','w') as f:
        json.dump(data,f)