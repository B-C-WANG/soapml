
from VDE.AtomSpace import  atom_index_trans_reverse
from soaplite import getBasisFunc, get_soap_locals, get_periodic_soap_locals
from ase import Atoms
import numpy as np
from collections import Counter

class SOAPTransformer():
    def __init__(self,encode_atom_cases,
                 n_max=8,l_max=8,r_cut=15.0):
        '''


           encode_atom_cases: if sample contrains atom H O C, but other dataset
           contains atom H O C and Pt, to make dataset same, use H O C and Pt
           Pt will be an absent atom, its position will be set to <absent_atom_default_position>

           if <absent_atom_default_position> is None:
           then will set to [-10,-10,-10] to center_position,
           test shows that in that range, feature of the absent atom will be zero vector

           n_max, l_max : related to the basis number in soaplite, set it bigger will get larger vector




           '''
        self.n_max = n_max
        self.l_max = l_max
        self.r_cut = r_cut
        self.needed_atom_cases = encode_atom_cases




    def transform(self,coord_with_atom_case,center_position,absent_atom_default_position=None,relative_absent_position=False,debug=False,periodic=False,cell_info=None):
        '''


          coord_with_atom_case(sample): shape (-1,4)
           4: atom_case num and x y z, like H atom: [1,0,0,0], O atom: [8,1,1.2,3.0]

           center_position: the position we are interested, will calculate
           the local chemical environment in that position

            absent_default_position: if the atom of atom case not in dataset, will set to default_position
            if it is None, set position to [-10,-10,-10]  of center postition

            periodic: True if data have a periodic structure
            then cell info, a 3x3 array, should offer,
            this data can be found in POSCAR, if from VASP result




        '''
        #print("If met bug with set_cell, see doc in ase.Atom.set_cell")
        if periodic == True and cell_info is None:
            raise ValueError("Should give cell info if use periodic")


        self.sample = coord_with_atom_case
        assert self.sample.shape[1] == 4
        assert len(self.sample.shape) == 2

        if isinstance(center_position, list):
            if not isinstance(center_position[0], list):
                raise ValueError("Input center position must be a 2D array")
        elif len(center_position.shape) != 2:
                raise ValueError("Input center position must be a 2D array")

        sample = self.sample
        needed_atom_cases = self.needed_atom_cases

        if absent_atom_default_position is None:
            absent_atom_default_position = np.array(center_position)[0,:] - np.array([10,10,10]) # don't know why this is important set (10,10,10) is better than (30,30,30)
            if debug:
                print("Absent atom default position not set, use %s" % absent_atom_default_position)
        else:
            if relative_absent_position == False:
                absent_atom_default_position = absent_atom_default_position
            else:
                absent_atom_default_position = np.array(center_position)[0,:] - np.array(absent_atom_default_position)


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
        new_atom_coord = []
        final_atom_type = []
        for atom in atom_cases:

            string += atom_index_trans_reverse[atom]
            _t = atom_type_number[atom]
            # for atoms in atom case but not in dataset ....
            if _t == 0:  # that means absent
                if debug:
                    print("Absent atom: ",atom)
                string += "1"
                #  if a absent atom, set absent position -10,-10,-10
                new_atom_coord .append(np.array(absent_atom_default_position).reshape(-1, 3))
                final_atom_type.append(atom)
                #atom_coord = np.concatenate([atom_coord, ], axis=0)
            else:
                for i in range(sample.shape[0]):
                    if int(sample[i][0]) == atom:
                        new_atom_coord.append(sample[i][1:].reshape(-1, 3))
                        final_atom_type.append(atom)

                string += str(atom_type_number[atom])
        atom_coord = np.concatenate(new_atom_coord)
        if debug:
            print("Final atom_coord shape:",atom_coord.shape)
            print("Final atom type: ",final_atom_type)
        molecule_name = string
        if debug:
            print(molecule_name)
        atoms = Atoms(molecule_name, positions=atom_coord)
        if periodic == True:
            atoms.set_cell(cell_info)


        hpos = center_position


        n_max = self.n_max
        l_max = self.l_max
        r_cut = self.r_cut
        my_alphas, my_betas = getBasisFunc(r_cut, n_max)

        if periodic == False:
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
        else:
            x = get_periodic_soap_locals(atoms,
                hpos,
                my_alphas,
                my_betas,
                rCut=r_cut,
                NradBas=n_max,
                Lmax=l_max,
                crossOver=True)

        return x





