from getData import getVesselDataComponentsDiametersLengths

import glob
import os
import shutil

from joblib import Parallel, delayed
import multiprocessing

rootfolder = ""
folder_idx = [name for name in os.listdir(rootfolder) if os.path.isdir(os.path.join(rootfolder, name))]

for i in range(len(folder_idx)):    

    #  List all '.nii' files in directory
    path = os.path.join(rootfolder, folder_idx[i],'nii_files')

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    print(files)
    
    path_out = os.path.join(rootfolder, folder_idx[i],'Skeletons/')

    filelist = glob.glob(os.path.join(path_out, "*"))
    for f in filelist:
        shutil.rmtree(f)

    # Without parallel processing
    # j = 1
    # for img_name in files:
    #     print('%i : %s'%(i, img_name))
    #     print(img_name)
    #     print(os.path.join(rootfolder, folder_idx[j]))
    #     print(path)
    #     print(path_out)
    #     getVesselDataComponentsDiametersLengths(img_name, folder=os.path.join(rootfolder, folder_idx[j]), 
    #                                             path_in=path, path_out=path_out)
    #     j += 1
    
    # #  With parallel processing
    num_cores = multiprocessing.cpu_count() # use max number of cores
    squares = Parallel(n_jobs=num_cores, verbose=50)(delayed(
        getVesselDataComponentsDiametersLengths)(img_name, folder=os.path.join(rootfolder, folder_idx[i]), 
                                                  path_in=path, path_out=path_out)for img_name in files)

