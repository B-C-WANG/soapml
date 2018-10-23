# soapml
- get SOAP (Smooth Overlap of Atomic Positions) feature from [SOAPLite](https://github.com/SINGROUP/SOAPLite) and do machine learning
## SOAPTransformer
### init
- **encode\_atom\_cases** atom cases include, e.g. for H O and C, atom cases is [1,8,6]. If there's any atom cases not in dataset, its coordiante of that atom type will be <**absent\_atom\_default\_position**\>(set in .transform). 
- **n\_max, l\_max, r\_cut** are parameter of SOAPLite
### .transform
- **coord\_with\_atom\_case** shape (?,4), atom_cases and x y z, e.g. [1, 1.2, 1.3, 1.4].
- **center_position** a 2D array with shape (?,3), SOAPLite will calculate the local environment in of the center\_position as feature.
- **absent\_atom\_default\_position** if there's any atom not in **encode\_atom\_cases** (defind in init), its coordiante will be **absent\_atom\_default\_position**, if it is None, will be the first **center\_position** add [-10,-10,-10]
