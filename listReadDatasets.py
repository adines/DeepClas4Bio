import json

readDatasets=[{'name':'ReadDatasetFolders','description':'The images have to be organized in folders. Each folder must have the name of the class which the images belong.'}]

if __name__=="__main__":
    data={}
    data['type']='readDatasets'
    data['readDatasets']=readDatasets
    with open('data.json','w') as f:
        json.dump(data,f)