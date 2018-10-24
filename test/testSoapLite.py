from soaplite import getBasisFunc, get_soap_locals, get_periodic_soap_structure,get_periodic_soap_locals,get_soap_structure
from ase import Atoms
from ase.build import molecule
#-------------- Define structure -----------------------------------------------
atoms = molecule("H2O")
# use custom structure as follows:
atoms = Atoms('H5',  # if use H2O instead of H3, array shape changes
          positions=[[0, 0, 0],
                     [0, 0, 0],
                     [0,0,0],
                     [1,1,1],
                     [0,0,0]
                     ])

#-------------- Define positions of desired local environments ----------------
# the local environment in that position
hpos = [
    [0, 0, 0],
    [0, 0, 1]
]

#------------------ Basis function settings (rCut, N_max) ----------------------
n_max = 5
l_max = 5
r_cut = 10.0
my_alphas, my_betas = getBasisFunc(r_cut, n_max)

#--------- Get local chemical environments for each defined position -----------
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

xs = get_soap_structure(atoms,

    my_alphas,
    my_betas,
    rCut=r_cut,
    NradBas=n_max,
    Lmax=l_max,
    crossOver=True)
# xps = get_periodic_soap_structure(atoms,
#
#     my_alphas,
#     my_betas,
#     rCut=r_cut,
#     NradBas=n_max,
#     Lmax=l_max,
#     crossOver=True)
# xp = get_periodic_soap_locals(atoms,
#     hpos,
#     my_alphas,
#     my_betas,
#     rCut=r_cut,
#     NradBas=n_max,
#     Lmax=l_max,
#     crossOver=True)

print(x)
print(x.shape)

print(xs)
print(xs.shape)
print(xp)
print(xp.shape)
print(xps)
print(xps.shape)
# output shape (num_position, feature_num)