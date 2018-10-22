
from VDE.AtomSpace import  atom_index_trans_reverse
from soaplite import getBasisFunc, get_soap_locals
from ase import Atoms
import numpy as np
from collections import Counter

class SOAPTransformer():
    def __init__(self,encode_atom_cases,
                 n_max=5,l_max=5,r_cut=10.0):
        '''


           encode_atom_cases: if sample contrains atom H O C, but other dataset
           contains atom H O C and Pt, to make dataset same, use H O C and Pt
           Pt will be an absent atom, its position will be set to <absent_atom_default_position>

           if <absent_atom_default_position> is None:
           then will set to [-10,-10,-10] to center_position,
           test shows that in that range, feature of the absent atom will be zero vector

           '''
        self.n_max = n_max
        self.l_max = l_max
        self.r_cut = r_cut
        self.needed_atom_cases = encode_atom_cases




    def transform(self,coord_with_atom_case,center_position,absent_atom_default_position=None):
        '''


          coord_with_atom_case(sample): shape (-1,4)
           4: atom_case num and x y z, like H atom: [1,0,0,0], O atom: [8,1,1.2,3.0]

           center_position: the position we are interested, will calculate
           the local chemical environment in that position

        :return:
        '''

        self.sample = coord_with_atom_case
        assert self.sample.shape[1] == 4
        assert len(self.sample.shape) == 2

        if isinstance(center_position, list):
            if not isinstance(center_position[0], list):
                raise ValueError("Input center position must be a 2D array")
        elif isinstance(center_position, np.array):
            if len(center_position.shape != 2):
                raise ValueError("Input center position must be a 2D array")

        sample = self.sample
        needed_atom_cases = self.needed_atom_cases

        if absent_atom_default_position is None:
            absent_atom_default_position = np.array(center_position)[0,:] - np.array([10,10,10])
            print("Absent atom default position not set, use %s" % absent_atom_default_position)
        else:
            absent_atom_default_position = absent_atom_default_position

        # sort atom types to match the sort of ase atom name
        sample = np.sort(sample, axis=0)
        atom_type = sample[:, 0].astype('int')
        atom_coord = sample[:, 1:]
        # sort to make it same for atom types
        atom_cases = sorted(list(set(needed_atom_cases)))
        # if there's any atom not in needed, move to last
        atom_type_number = Counter(atom_type)
        have_atom = list(atom_type_number.keys())
        absent = []
        for i in atom_cases:
            if i not in have_atom:
                absent.append(i)
        for i in absent:
            atom_cases.remove(i)
        atom_cases.extend(absent)

        string = ""
        for atom in atom_cases:
            string += atom_index_trans_reverse[atom]
            _t = atom_type_number[atom]
            if _t == 0:  # that means absent
                string += "1"
                #  if a absent atom, set absent position -10,-10,-10
                atom_coord = np.concatenate([atom_coord, np.array(absent_atom_default_position).reshape(-1, 3)], axis=0)
            else:
                string += str(atom_type_number[atom])
        molecule_name = string
        print(molecule_name)
        atoms = Atoms(molecule_name, positions=atom_coord)

        hpos = [
            center_position
        ]

        n_max = self.n_max
        l_max = self.l_max
        r_cut = self.r_cut
        my_alphas, my_betas = getBasisFunc(r_cut, n_max)

        x = get_soap_locals(
            atoms,
            hpos,
            my_alphas,
            my_betas,
            rCut=r_cut,
            NradBas=n_max,
            Lmax=l_max,
            crossOver=True
        )

        return x





