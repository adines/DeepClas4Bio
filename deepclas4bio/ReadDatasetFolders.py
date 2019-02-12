from deepclas4bio import IReadDataset
import glob
import os
import numpy as np

class ReadDatasetFolders(IReadDataset.IReadDataset):
    def readDataset(self,path,pathLabels):
        folders=glob.glob(path+os.sep+'*')
        images=[]
        groundTruth=[]
        labels = np.loadtxt(pathLabels, str, delimiter='\n')
        for folder in folders:
            pos=folder.rfind(os.sep)
            imagesFolder=glob.glob(folder+os.sep+'*')
            for image in imagesFolder:
                images.append(image)
                groundTruth.append(list(labels).index(folder[pos+1:]))
        return [images,groundTruth]
