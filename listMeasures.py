import Measures
import json

def listMeasures():
    bm=Measures.binaryMeasures
    nbm=Measures.noBinaryMeasures
    return list(set().union(bm,nbm))


if __name__=="__main__":
    data={}
    data['type']='measures'
    data['measures']=listMeasures()
    with open('data.json','w') as f:
        json.dump(data,f)