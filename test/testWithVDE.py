
from VDE.VASPMoleculeFeature import VASP_DataExtract
from soapml.SOAPTransformer import SOAPTransformer

test = VASP_DataExtract("/home/yb/Desktop/Dataset/碳纳米管掺杂/整理后/5-5/b/h/3")
a = test.get_output_as_atom3Dspace()
coord, energy, atom_cases = a.generate_data()
atom_cases = set(atom_cases)
atom_cases.update([5, 7,8])  # add B and N
atom_cases.remove(1) # remove H
#atom_cases.remove(6) # remove C
transformer = SOAPTransformer(encode_atom_cases=atom_cases)

for i in range(coord.shape[0]):
    sample = coord[-1, :, :]
    x = transformer.transform(sample, [[0, 0, 0]])
    print(x.shape)
    print(x[...,0])
    print(...)
    exit()

