from VDE.VASPMoleculeFeature import  VASP_DataExtract
from soapml.SOAPTransformer import  SOAPTransformer
import pandas as pd
import numpy as np
import pickle as pkl
import tqdm



class Dataset(object):
    '''
    we will use all steps in a Vasp dir, rather than final one



    '''

    def __init__(self,coord,energy,box_tensor=None,description=""):


        self.coord = coord
        self.energy = energy
        self.box_tensor = box_tensor
        self.description = description
        self.repeat_count = 0
        self.repeated = False





    @staticmethod
    def from_vasp_dir_and_energy_table(vasp_dir_table,descriprion=""):
        '''
        give a .xlsx/.csv file name, content is like:
        Vasp Dirs | slab energy
        Pt_OH1    | 1.0
        Pt_OH2    | 2.0
        Pt_OH3    | 3.0
        '''
        error_info = "input table must be a .xlsx/.csv file name, use generate_vasp_dir_energy_table to get one table."
        assert  isinstance(vasp_dir_table,str), error_info
        if vasp_dir_table.endswith(".xlsx"):
            data = pd.read_excel(vasp_dir_table)
        elif vasp_dir_table.endswith(".csv"):
            data = pd.read_csv(vasp_dir_table)
        else:
            raise ValueError(error_info)

        try:
            vasp_dirs = data["Vasp Dirs"]
            slab_energy = data["slab energy"]
        except KeyError:
            raise ValueError("Input table must contain Vasp Dirs and slab energy, use generate_vasp_dir_energy_table to get one correct table.")
        return Dataset.from_vasp_dir_and_energy_list(vasp_dirs,slab_energy,descriprion)

    @staticmethod
    def from_vasp_dir_and_energy_list(vasp_dirs,slab_energy,description=""):


        coordinate = []
        energy = []
        box_tensor = []
        print("\nNow extracting data from Vasp ... ")
        for i in tqdm.trange(len(vasp_dirs)):
            vasp_dir = vasp_dirs[i]
            vde = VASP_DataExtract(vasp_dir)
            coord,_,_ = vde.get_output_as_atom3Dspace().generate_data()
            e = vde.get_energy_info()
            vde.get_box_tensor_and_type_info()
            _box_tensor = vde.box_tensor_info
            _t = []
            for j in range(len(e)):
                _t.append(e[j+1])
            e = _t
            coordinate.append(coord)
            energy.append(np.array(e)-float(slab_energy[i]))
            box_tensor.append(_box_tensor)
        return Dataset(coordinate,energy,box_tensor,description)




    @staticmethod
    def from_coordinate_and_energy_array(coordinate,energy,box_tensor=None,description=""):
        '''
        e.g.
        (two group, first group contains two sample, each sample is two H
        second group contains one sample, two H and a He atom)
        coord = [
        np.array(
        [
            [
                [1, 0.0, 1.0, 2.0],
                [1, 0.0, 2.0, 3.0]
            ],
            [
                [1, 0.0, 1.0, 2.0],
                [1, 0.0, 2.0, 3.0]
            ]
        ]
        ),

        np.array([
            [1, 1.0, 1.0, 2.0],
            [1, 1.0, 2.0, 3.0],
            [2, 1.0, 2.0, 3.0]
        ]
        ).reshape(1,-1,4)
        ]
        energy = [np.array([0.0,1.0]),np.array([0.5])]
        (energy of two groups)

        box_tensor: a 3x3 matrix, contain cell param
        '''

        error_info_coord = "Coordinate must be a LIST of ARRAY ARRAY which every array has shape (n_samples, n_atoms, 4), 4 is atom_index, x, y, z."
        error_info_energy = "Energy must be a LIST of ARRAY which every array has shape (n_samples, 1), every n_samples MUST be same as n_samples in coordinate"
        assert  isinstance(coordinate,list),error_info_coord
        assert  isinstance(energy, list),error_info_energy
        assert isinstance(coordinate[0],np.ndarray), error_info_coord
        assert isinstance(energy[0], np.ndarray), error_info_energy
        assert len(coordinate) == len(energy), "Coordinate num should equal to energy num"
        if box_tensor is not None:
            assert isinstance(box_tensor, list) , "Box tensor must be a List of 3x3 array, "
            assert len(box_tensor) == len(coordinate) == len(energy), "Coordinate, energy and box_tensor list should be with same length"
        else:
            assert  len(coordinate) == len(
                energy), "Coordinate and energy list should be with same length"
        for i in range(len(coordinate)):
            c_shape = coordinate[i].shape
            e_shape = energy[i].shape
            assert c_shape[0] == e_shape[0], "In %sth sample, coordinate and energy num not match, which has %s %s"%(i,c_shape,e_shape)
        return Dataset(coordinate,energy,box_tensor,description)

    @staticmethod
    def generate_vasp_dir_energy_table(vasp_dir,to_csv=False):
        '''
        give a dir contains many vasp dirs, like the dir named Vasp, that contrains: Pt_OH1, Pt_OH2, Pt_OH3 ...
        and generate a excel/csv table like:
        Vasp Dirs | slab energy
        Pt_OH1    | 0
        Pt_OH2    | 0
        Pt_OH3    | 0
        and then you need to fill the slab energy
        '''
        vasp_dirs = VASP_DataExtract.get_all_VASP_dirs_in_dir(vasp_dir)
        energy = [0 for _ in range(len(vasp_dirs))]
        t = np.array([vasp_dirs,energy])
        t = t.T

        data = pd.DataFrame(t)
        data.stack(level=-1)
        data.columns = ["Vasp Dirs","slab energy"]
        if to_csv:
            data.to_csv("vasp_dir_path_energy_table.csv", index=False)
        else:
            data.to_excel("vasp_dir_path_energy_table.xlsx",index=False)


    def sample_filter(self,ratio=0.15):
        '''
        we think that first 15% sample of Vasp is not stable,
        they have a big influence on model,
        choose to keep or not

        '''
        new_coord = []
        new_energy = []
        print("\nNow filter samples ...")
        for i in tqdm.trange(len(self.coord)):
            c = self.coord[i]
            e = self.energy[i]
            split_point = int(c.shape[0] * ratio)
            new_coord.append(c[split_point:,:,:])
            new_energy.append(e[split_point:])
        self.coord = new_coord
        self.energy = new_energy

    def apply_period(self,direction,repeat_count=1):
        '''

        for the original structure, the atom on the edge will
        have a big difference to atom on the center
        so we need to repeat the period structure



        all sample will apply one direction period
        but box_tensor can be different!

        box_tensor is a 3x3 array, contains:
        box_tensor[0]: VectorA
        box_tensor[1]: VectorB
        box_tensor[2]: VectorC
        direction: 0-A 1-B 2-C

        repeat_count:
        if 1: repeat once: add Vector * 1 and Vector * -1 to original coord
        if 2: add Vector * 2, 1, -1, -2 to original coord

        e.g. Vector A is [0,1,0.5]
        atom 1 is [0,0,0]
        when repeat = 0

        '''
        if repeat_count == 0:
            return
        else:
            self.repeated = True
        self.repeated_coord = []
        self.repeat_count = repeat_count # this param is need for predict!
        print("\nNow applying period ...")
        for i in tqdm.trange(len(self.coord)):
            c = self.coord[i]
            bt = self.box_tensor[i]
            # 0 represents atom cases add 0 to atom cases
            change = np.array([0,bt[direction][0],bt[direction][1],bt[direction][2]]).astype("float32")
            repeat = list(range(-repeat_count,repeat_count+1))
            repeat.remove(0)
            repeat.insert(0,0)

            new_c = []
            for j in repeat:
                offset = change * float(j)
                new_c.append(c+offset)
            new_c = np.concatenate(new_c,axis=1)
            self.repeated_coord.append(new_c)

    def soap_encode(self,center_atom_cases,encode_atom_cases,n_max=8,l_max=8,r_cut=15.0,absent_atom_default_position=None):

        self.soap_transformer = SOAPTransformer(encode_atom_cases=encode_atom_cases, n_max=n_max, l_max=l_max,
                                           r_cut=r_cut)
        x_feature = []
        y = []
        print("\nNow soap encoding ...")
        for i in tqdm.trange(len(self.coord)):
            for j in range(self.coord[i].shape[0]):
                if self.repeated:
                    coord_ = self.repeated_coord[i][j,:,:]
                else:
                    coord_ = self.coord[i][j,:,:]
                aim_atom_pos = []
                for atom in center_atom_cases:
                    # use not repeated (center) coord
                    c = self.coord[i][j,:,:]

                    aim_atom_pos.append(c[c[:,0]==atom])

                aim_atom_pos = np.concatenate(aim_atom_pos,axis=0)
                center_pos = np.mean(aim_atom_pos,axis=0)[1:].reshape(1,-1)
                x_feature.append(self.soap_transformer.transform(coord_,center_position=center_pos,periodic=False,absent_atom_default_position=absent_atom_default_position))
                y.append(self.energy[i][j])
        x = np.concatenate(x_feature,axis=0)
        y = np.array(y)
        self.datasetx = x
        self.datasety = y

    def save(self,filename="dataset.smld"):
        with open(filename, "wb") as f:
            pkl.dump(self,f)

    @staticmethod
    def load(filename="dataset.smld"):
        with open(filename,"rb") as f:
            return pkl.load(f)

    def __str__(self):
        return self.description






def tst_give_out_table():
    Dataset.generate_vasp_dir_energy_table("C:\\Users\wang\Desktop\运行结果")
def tst_read_table():
    Dataset.from_vasp_dir_and_energy_table("vasp_dir_path_energy_table.xlsx")
def tst_from_coord_and_energy():
    coord = [
        np.array(
        [
            [
                [1, 0.0, 1.0, 2.0],
                [1, 0.0, 2.0, 3.0]
            ],
            [
                [1, 0.5, 1.5, 2.5],
                [1, 0.5, 2.5, 3.5]
            ]
        ]
        ),

        np.array([
            [1, 1.0, 1.0, 2.0],
            [1, 1.0, 2.0, 3.0],
            [2, 1.0, 2.0, 3.0]
        ]
        ).reshape(1,-1,4)
    ]
    energy = [np.array([0.0,1.0]),np.array([0.5])]
    box_tensor = [
        np.array([
            [.5,.5,1],
            [0,1,0],
            [0,0,1]
        ])

    ] * 2
    dataset = Dataset.from_coordinate_and_energy_array(coordinate=coord,energy=energy,box_tensor=box_tensor,
                                                       description="This is a example")
    assert isinstance(dataset, Dataset)
    dataset.apply_period(0, repeat_count=3)
    dataset.soap_encode(center_atom_cases=[1,2],encode_atom_cases=[1])
    print(dataset.datasetx.shape)
    print(dataset.datasety.shape)
    dataset.save()
    print(Dataset.load())


if __name__ == '__main__':
    #tst_give_out_table()
    #tst_read_table()
    tst_from_coord_and_energy()

