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

    def keep_data_larger_than(self,y):
        def condition(data):
            return True if data > y else False
        self.keep_data_in_condition(condition)

    def keep_data_smaller_than(self,y):
        def condition(data):
            return True if data < y else False
        self.keep_data_in_condition(condition)


    def keep_data_in_condition(self,condition):
        # function as param
        y = self.y
        x = self.x
        y = y.reshape(-1, 1)
        print(x.shape, y.shape)
        dataset = np.concatenate([x, y], axis=1)
        new_data = []
        for i in range(dataset.shape[0]):

            if condition(dataset[i, -1] ):
                new_data.append(dataset[i, :])
        new_dataset = np.array(new_data)
        x = new_dataset[:, :-1]
        y = new_dataset[:, -1]
        self.x = x
        self.y = y


    def fit_gbr(self,test_split_ratio=0.3,n_estimators=1000):
        self.model = GBRFIE(self.x,self.y,test_split_ratio=test_split_ratio)
        self.model.fit(n_estimators)
        self.error = self.model.show_pred_train_test(plot_fig=True)
        print("Got error: %s"%self.error)

    def __str__(self):
        string = self.description
        string += "\nDataset X shape: %s"% str(self.x.shape)
        return string

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


