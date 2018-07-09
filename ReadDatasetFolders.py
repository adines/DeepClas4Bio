import IReadDataset
import glob
import os

class ReadDatasetFolders(IReadDataset.IReadDataset):
    def readDataset(self,path):
        folders=glob.glob(path+os.sep+'*')
        images=[]
        groundTruth=[]
        for folder in folders:
            pos=folder.rfind(os.sep)
            imagesFolder=glob.glob(folder+os.sep+'*')
            for image in imagesFolder:
                images.append(image)
                groundTruth.append(folder[pos+1:])
        return [images,groundTruth]
