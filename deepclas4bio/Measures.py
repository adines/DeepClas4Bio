from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
import numpy as np

binaryMeasures=['accuracy','precision','recall','f1','jaccardIndex','matthewsCorrelation','auroc']
noBinaryMeasures=['accuracy','rank5']

def accuracy(predictions,trueLabels,sort=True):
    i=0
    positive=0
    for prediction in predictions:
        prediction = np.squeeze(prediction)
        if sort:
            prediction = np.argsort(prediction)[::-1]
        if prediction[0]==trueLabels[i]:
            positive+=1
        i+=1
    return positive/i

def rank5(predictions,trueLabels,sort=True):
    i = 0
    positive = 0

    for prediction in predictions:
        prediction = np.squeeze(prediction)
        if sort:
            prediction = np.argsort(prediction)[::-1]
        if trueLabels[i] in prediction[:5]:
            positive+=1
        i+=1
    return positive/i

def precision(predictions,trueLabels):
    predictionsBinary=[]
    for prediction in predictions:
        predictionsBinary.append(prediction[0])
    return precision_score(trueLabels,predictionsBinary)

def recall(predictions,trueLabels):
    predictionsBinary = []
    for prediction in predictions:
        predictionsBinary.append(prediction[0])
    return recall_score(trueLabels,predictionsBinary)

def f1(predictions,trueLabels):
    predictionsBinary = []
    for prediction in predictions:
        predictionsBinary.append(prediction[0])
    return f1_score(trueLabels,predictionsBinary)

def jaccardIndex(predictions,trueLabels):
    predictionsBinary = []
    for prediction in predictions:
        predictionsBinary.append(prediction[0])
    return jaccard_similarity_score(trueLabels,predictionsBinary)

def matthewsCorrelation(predictions,trueLabels):
    predictionsBinary = []
    for prediction in predictions:
        predictionsBinary.append(prediction[0])
    return matthews_corrcoef(trueLabels,predictionsBinary)

def auroc(predictions,trueLabels):
    predictionsBinary = []
    for prediction in predictions:
        predictionsBinary.append(prediction[0])
    return roc_auc_score(trueLabels,predictionsBinary)