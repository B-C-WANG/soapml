from VDE.VASPMoleculeFeature import  VASP_DataExtract
from soapml.SOAPTransformer import  SOAPTransformer
import pandas as pd
import numpy as np
import pickle as pkl
import tqdm
import warnings



class Dataset(object):
    '''
    we will use all steps in a Vasp dir, rather than final one



    '''

    def __init__(self,coord,energy=None,box_tensor=None,only_x=False,description=""):


        self.coord = coord
        self.energy = energy
        self.box_tensor = box_tensor
        self.description = description
        self.repeat_config = []
        self.repeated = False
        self.repeated_coord = None
        self.only_x = only_x




    @staticmethod
    def from_vasp_dir_and_energy_table(vasp_dir_table,only_x=False,description=""):
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
            if only_x == True:
                vasp_dirs = data["Vasp Dirs"]
                slab_energy = None
                return Dataset.from_vasp_dir_and_energy_list(vasp_dirs, slab_energy, only_x=True,description=description)
            else:
                vasp_dirs = data["Vasp Dirs"]
                slab_energy = data["slab energy"]
                return Dataset.from_vasp_dir_and_energy_list(vasp_dirs, slab_energy, only_x=False,description=description)
        except KeyError:
            raise ValueError("Input table must contain Vasp Dirs and slab energy, use generate_vasp_dir_energy_table to get one correct table.")


    @staticmethod
    def from_vasp_dir_and_energy_list(vasp_dirs,slab_energy=None,final_ads_energy=None,only_x=False,description=""):
        '''

        if use slab energy, y will be energy of every step - slab energy
        if use final_energy, y will be energy of every step - energy of final step + final energy


        '''
        if only_x==False:
            assert slab_energy != None or final_ads_energy != None, "At least feed one reference energy"
            if (slab_energy != None and final_ads_energy != None):
                raise ValueError("Can only feed one type of energy!")
            if slab_energy != None:use_slab_energy = True
            else:use_slab_energy = False

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
            if only_x == False:
                if use_slab_energy:
                    energy.append(np.array(e)-float(slab_energy[i]))
                else:
                    energy.append(np.array(e)-float(e[-1]) + float(final_ads_energy[i]))
            box_tensor.append(_box_tensor)
        if only_x == True:
            return Dataset(coordinate,energy,box_tensor,only_x=True,description=description)

        else:
            return Dataset(coordinate,energy,box_tensor,only_x=False,description=description)




    @staticmethod
    def from_coordinate_and_energy_array(coordinate,energy=None,box_tensor=None,only_x=False,description=""):
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
        if only_x == False:assert  isinstance(energy, list),error_info_energy
        assert isinstance(coordinate[0],np.ndarray), error_info_coord
        if only_x == False:
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
            if only_x == False:
                e_shape = energy[i].shape
                assert c_shape[0] == e_shape[0], "In %sth sample, coordinate and energy num not match, which has %s %s"%(i,c_shape,e_shape)
        if only_x == True:
            return Dataset(coordinate,energy,box_tensor,only_x=True,description=description)
        else:
            return Dataset(coordinate,energy,box_tensor,only_x=False,description=description)

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
            split_point = int(c.shape[0] * ratio)
            new_coord.append(c[split_point:,:,:])
            if self.only_x == False:
                e = self.energy[i]
                new_energy.append(e[split_point:])
        self.coord = new_coord
        self.energy = new_energy

    def give_a_sample_from_dataset(self,sample_group_index,sample_index,use_repeated=True):
        if self.repeated_coord == True and use_repeated== True:
            return self.repeated_coord[sample_group_index][sample_index,:,:]
        else:
            if use_repeated == True:
                warnings.warn("Haven't apply period, can not export repeated structure.")
            return self.coord[sample_group_index][sample_index,:,:]

    @staticmethod
    def from_slab_and_center_position(slab_structure, center_position,box_tensor=None):
        slab_error_info = "Input slab should be an array of (n_atoms, 4), but shape is %s"
        center_position_info = "Input center_position should be an array of (n, 3), but shape is %s"

        assert len(slab_structure.shape) == 2 and slab_structure.shape[1] == 4, slab_error_info % (slab_structure.shape)
        assert len(center_position.shape) == 2 and center_position.shape[1] == 3, center_position_info % (
            center_position.shape)

        slab_structure = [slab_structure.reshape(1,slab_structure.shape[0],slab_structure.shape[1])]

        return Dataset(coord=slab_structure,box_tensor=box_tensor,only_x=True)

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
        if self.repeated_coord is None:
            self.repeated_coord = []
            coord = self.coord
        else:
            coord = self.repeated_coord
            self.repeated_coord = []
        self.repeat_config.append([direction,repeat_count]) # this param is need for predict!
        print("\nNow applying period ...")
        for i in tqdm.trange(len(coord)):
            c = coord[i]

            if len(c.shape) == 2:
                c = c.reshape(1, c.shape[0], c.shape[1])
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

    def atom_dependent_soap_encode(self,center_atom_cases,encode_atom_cases,n_max=8,l_max=8,r_cut=15.0,absent_atom_default_position=[10,10,10],relative_absent_position=True):
        '''
        every atom of case in center_atom_cases will be encode
        result:
        list of dict:
        list length = num of sample group
        dict, key : atom cases
        value: (n_sample, n_atom of case (same as key), feature)

        '''

        self.encode_atom_cases = encode_atom_cases
        self.n_max = n_max
        self.l_max = l_max
        self.r_cut = r_cut
        self.absent_atom_default_position = absent_atom_default_position
        self.relative_absent_position = relative_absent_position

        self.soap_transformer = SOAPTransformer(encode_atom_cases=encode_atom_cases, n_max=n_max, l_max=l_max,
                                                r_cut=r_cut)
        result  =[]
        for i in tqdm.trange(len(self.coord)):
            encode_result = {}
            for atom_case in center_atom_cases:
                encode_result[atom_case] = []
            for j in range(self.coord[i].shape[0]):
                if self.repeated:
                    coord_ = self.repeated_coord[i][j,:,:]
                else:
                    coord_ = self.coord[i][j,:,:]


                for atom_case in center_atom_cases:
                    center_coord = self.coord[i][j, :, :]
                    encode_input = center_coord[center_coord[:, 0] == atom_case][:, 1:]
                    if encode_input.shape[0] == 0: # if no such atom, dict do not add
                        encode_output = 0
                    # the shape[0] should be atom number, even it is 1 atom
                    else:


                        encode_output = self.soap_transformer.transform(coord_,center_position=encode_input,
                                                                 periodic=False,
                                                                 absent_atom_default_position=absent_atom_default_position,

                                                                 relative_absent_position=relative_absent_position)

                    feature_num = encode_output.shape[1]

                encode_result[atom_case].append(encode_output)

                # if some atom not in center_atom_cases, make it zero vector, length == feature_num
            encode_atoms = list(encode_result.keys())
            for need_atoms in center_atom_cases:
                #print(need_atoms,encode_atoms)


                encode_result[need_atoms] = np.array(encode_result[need_atoms])
                if len(encode_result[need_atoms].shape) == 1: # it

                    print("No atom %s, set to zero feature" % need_atoms)
                    encode_result[need_atoms] = np.zeros(shape=(self.coord[i].shape[0],1,feature_num))
            result.append(encode_result)

        self.datasetx = result
        self.datasety = self.energy # do not need any transform to y


    def soap_encode(self,encode_atom_cases,center_atom_cases=None,center_position=None,n_max=8,l_max=8,r_cut=15.0,absent_atom_default_position=[10,10,10],relative_absent_position=True):

        if center_atom_cases is None and center_position is None:
            raise ValueError("At least set one of center_atom_cases, center_position")
        if center_position is not None and center_atom_cases is not None:
            raise ValueError("Can not set Both center_atom_cases and center_position, what do you want to do?")
        self.encode_atom_cases = encode_atom_cases
        self.n_max = n_max
        self.l_max = l_max
        self.r_cut = r_cut
        self.absent_atom_default_position = absent_atom_default_position
        self.relative_absent_position= relative_absent_position

        self.soap_transformer = SOAPTransformer(encode_atom_cases=encode_atom_cases, n_max=n_max, l_max=l_max,
                                           r_cut=r_cut)
        x_feature = []
        y = []
        print("\nNow soap encoding ...")

        for i in tqdm.trange(len(self.coord)): # TODO: use parallel python to speed up !

            for j in range(self.coord[i].shape[0]):
                if self.repeated:

                    coord_ = self.repeated_coord[i][j,:,:]
                else:
                    coord_ = self.coord[i][j,:,:]


                if center_position is None:
                    aim_atom_pos = []
                    for atom in center_atom_cases:
                        # use not repeated (center) coord
                        c = self.coord[i][j, :, :]

                        aim_atom_pos.append(c[c[:, 0] == atom])

                    aim_atom_pos = np.concatenate(aim_atom_pos, axis=0)
                    center_pos = np.mean(aim_atom_pos,axis=0)[1:].reshape(1,-1)
                else:
                    center_pos = center_position
                x_feature.append(self.soap_transformer.transform(coord_,center_position=center_pos,
                                                                 periodic=False,
                                                                 absent_atom_default_position=absent_atom_default_position,
                                                                 relative_absent_position=relative_absent_position))
                if self.only_x == False:
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

