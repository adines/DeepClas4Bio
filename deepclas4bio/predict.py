from deepclas4bio import DatasetManager
from deepclas4bio.PredictorFactory import PredictorFactory

def predict(image,framework,model):
    predictor_factory=PredictorFactory()
    modelo=predictor_factory.getPredictor(framework,model)
    return modelo.predict(image)

def predictBatch(images,framework,model,batch=64):
    predictor_factory=PredictorFactory()
    modelo=predictor_factory.getPredictor(framework,model)
    dataManager= DatasetManager.DatasetManager(images, batch=batch)
    predictions=[]
    while(dataManager.hasNextBach()):
        batchImages=dataManager.nextBatch()
        prediction=modelo.predictBatch(batchImages)
        predictions+=prediction
    results=[]
    for p in predictions:
        results.append(modelo.model.postProcessor(p))
    return results