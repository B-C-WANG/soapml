
from VDE.VASPMoleculeFeature import VASP_DataExtract
from soapml.SOAPTransformer import SOAPTransformer

test = VASP_DataExtract("/home/yb/Desktop/Dataset/2")
a = test.get_output_as_atom3Dspace()
coord, energy, atom_cases = a.generate_data()
atom_cases.extend([5, 7])  # add B and N
transformer = SOAPTransformer(encode_atom_cases=atom_cases)

for i in range(coord.shape[0]):
    sample = coord[0, :, :]
    x = transformer.transform(sample, [[0, 0, 0]])
    print(x.shape)


