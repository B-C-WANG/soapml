from soapml.Dataset import Dataset
from MLT.Regression.GBR_FeatureImportanceEstimater import GBRFIE
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
class Model():
    def __init__(self,dataset):
        assert  isinstance(dataset,Dataset), "Input must be soapml.Dataset"
        self.x = dataset.datasetx
        self.y = dataset.datasety
        self.description = dataset.description
        self.repeat_config = dataset.repeat_config
        self.box_tensor = dataset.box_tensor
        self.model = None
        self.transformer = dataset.soap_transformer
        self.encode_atom_cases = dataset.encode_atom_cases
        self.n_max = dataset.n_max
        self.l_max = dataset.l_max
        self.r_cut = dataset.r_cut
        self.absent_atom_default_position = dataset.absent_atom_default_position
        self.relative_absent_position = dataset.relative_absent_position

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

    @staticmethod
    def make_grid(
                  x_res,
                  y_res,
                  z_res,
                  x_max,
            x_min,
            y_max,
            y_min,
            z_max,
            z_min
                  ):
        pass




    def encode(self,dataset,center_atom_cases=None,center_position=None,sample_filter_ratio=0):
        '''
        encode the same way as the dataset in Model(Dataset)
        the information of encoding of trainset have passed from
        dataset into Model
        dataset can contain y or not
        '''
        assert isinstance(dataset, Dataset)
        if sample_filter_ratio >= 0.001:
            dataset.sample_filter(ratio=sample_filter_ratio)
        if len(self.repeat_config) >= 1:
            for i in self.repeat_config:
                dataset.apply_period(i[0],i[1])

        dataset.soap_encode(center_atom_cases=center_atom_cases,
            center_position=center_position,
                            encode_atom_cases=self.encode_atom_cases,
                            absent_atom_default_position=self.absent_atom_default_position,
                            relative_absent_position=self.relative_absent_position)
        return dataset

    def predict_and_validate(self,dataset):
        x=dataset.datasetx
        try:
            true_y = dataset.datasety
            if true_y is None:
                raise ValueError("No y in dataset, can not validate!")
        except:
            raise ValueError("No y in dataset, can not validate!")
        pred_y = self.model.model.predict(x)



        plt.scatter(pred_y, true_y)
        plt.show()

        error = np.mean(np.abs(pred_y - true_y))

        print("Validate error: ",error)


    def predict(self,dataset):
        try:
            x = dataset.datasetx
        except:
            raise ValueError("No data in dataset, please encode")
        return self.model.model.predict(x)




    def fit_gbr(self,test_split_ratio=0.3,n_estimators=1000,shuffle=True):
        self.model = GBRFIE(self.x,self.y,test_split_ratio=test_split_ratio,shuffle=shuffle)
        self.model.fit(n_estimators)
        self.error = self.model.show_pred_train_test(plot_fig=True)
        print("Got error: %s"%self.error)

    def __model_predict(self,x):
        return self.model.model.predict(x)

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


