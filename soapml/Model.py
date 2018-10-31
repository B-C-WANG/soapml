from soapml.Dataset import Dataset
from MLT.Regression.GBR_FeatureImportanceEstimater import GBRFIE

class Model():
    def __init__(self,dataset):
        assert  isinstance(dataset,Dataset), "Input must be soapml.Dataset"
        self.x = dataset.datasetx
        self.y = dataset.datasety
        self.description = dataset.description
        self.repeat_count = dataset.repeat_count
        self.box_tensor = dataset.box_tensor
        self.model_name = model_name

    def fit_gbr(self,test_split_ratio=0.3,n_estimators=1000):
        self.model = GBRFIE(self.x,self.y,test_split_ratio=test_split_ratio)
        self.model.fit(n_estimators)
        self.error = model.show_pred_train_test(plot_fig=True)
        print("Got error: %s"%self.error)


    @staticmethod
    def load(filename="soapModel.smlm"):
        with open(filename,"rb") as f:
            return pkl.load(f)

    def save(self,filename="soapModel.smlm"):
        with open(filename, "wb") as f:
            pkl.dump(self,f)





def tst():
    dataset = Dataset.load()
    model = Model(dataset)


