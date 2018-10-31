# soapml is under development
## feature:

### input contains many structure(of x y z or vasp dirs) and its adsorption energy, its recommended that you put many stucture of adsorbate+surface, and give energy of surface, since we will use energies of every structure rather than the most stable structure of adsorbate+surface, but energy of surface should be optimized.   
###input in three ways: 

- give a dir containing many vasp dirs, these dirs have surface structure and absorbates. for each dir, give out the energy without adsorbates(slab energy). e.g. dirs of OH on N doped carbon, OH on B doped carbon, and give slab energy of optimized N doped carbon and B doped carbon, you can use a script to generate table and fill the table with slab energy
- give a dir contraining many vasp dirs, and corresponding slab vasp dir, e.g. give OH + N doped carbon dir, and give a slab N doped carbon dir. Programme will extract slab energy from the second dir to calculate adsorption energy of OH
- give a array with shape （n_sample, 4)，contains samples with [atom_index, x, y, z]， and give absorption energy of every sample


# soapml
- get SOAP (Smooth Overlap of Atomic Positions) feature from [SOAPLite](https://github.com/SINGROUP/SOAPLite) and do machine learning


## SOAP "Probe" 
- prepare a surface and an adsorbate (e.g. H) as "Probe".
- get energy of surface without probe Es, move probe on the surface, get energy list Ep_l. (from DFT calculation...)
- now you have structures of different probe position on surface as X, Ep_l - Es as y
- use SOAPTransformer to transform X to features, the position of probe is as center position
- use ML models, like Gradient Boost Regression. Use soap features and y to train the model, save the model
- give a new surface, give many center positions, use SOAPTransformer to get features, and used saved gbr model to give energy.
- plot out.
- e.g. use atom H as probe, nanotube as surface, calculate  absorbation energy of H, got dataset, and use model to predict energy on a plane on the center of nanotube(red is high energy, blue is low energy):  
![](https://i.imgur.com/CGqdKuM.png)


## SOAPTransformer
### init
- **encode\_atom\_cases** atom cases include, e.g. for H O and C, atom cases is [1,8,6]. If there's any atom cases not in dataset, its coordiante of that atom type will be <**absent\_atom\_default\_position**\>(set in .transform). 
- **n\_max, l\_max, r\_cut** are parameter of SOAPLite
### .transform
- **coord\_with\_atom\_case** shape (?,4), atom_cases and x y z, e.g. [1, 1.2, 1.3, 1.4].
- **center_position** a 2D array with shape (?,3), SOAPLite will calculate the local environment in of the center\_position as feature.
- **absent\_atom\_default\_position** if there's any atom not in **encode\_atom\_cases** (defind in init), its coordiante will be **absent\_atom\_default\_position**, if it is None, will be the first **center\_position** add [-10,-10,-10]
