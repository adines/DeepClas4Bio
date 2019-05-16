
class DatasetManager:
    def __init__(self,images,batch=64):
        self.images=images
        self.batch=batch
        self.batchNumber=0
        self.numImages=len(self.images)

    def hasNextBach(self):
        return self.batchNumber*self.batch<self.numImages

    def nextBatch(self):
        result= self.images[self.batchNumber*self.batch:(self.batchNumber+1)*self.batch]
        self.batchNumber+=1
        return result