import os
import glob
import laspy as lp
import numpy as np

def convert_single_file_xyzrgb(las_fn, npy_fn):
    cloud = lp.read(las_fn)
    np.save(npy_fn, np.stack((cloud.x, cloud.y, cloud.z, cloud.red, cloud.green, cloud.blue)).transpose())
    

def convert_single_file_xyz(las_fn, npy_fn):
    cloud = lp.read(las_fn)
    np.save(npy_fn, np.stack((cloud.x, cloud.y, cloud.z)).transpose())

    
def debug_rgb(las_fn):
    cloud = lp.read(las_fn)

    length = len(cloud)
    print('Total lines of {}: '.format(las_fn), length)
    
    for i, (x, y, z, r, g, b) in enumerate(zip(cloud.x, cloud.y, cloud.z, cloud.red, cloud.blue, cloud.green)):
        if(i % 50000 == 0):
            print("Current file position: " + str(i))

        print("Original RGB value " + "{} {} {}".format(r,g,b) + " were converted to " + "{:e}".format((int(r)<<16) + (int(g)<<8) + int(b)))
            
    
def Main():
    import pdb
    dataset = 'sanlim_las' #sys.argv[1]
    las_dir = os.path.join('.', dataset)
    las_fns = glob.glob(os.path.join(las_dir, '*/*.*'))
    
    pcd_dir = os.path.join('sanlim_npy')
    labels = os.listdir(las_dir)
    os.makedirs(pcd_dir, exist_ok=True)
    for cls in labels:
        os.makedirs(os.path.join(pcd_dir, cls), exist_ok=True)
    
    for las_fn in las_fns:
        tar_fn = os.path.join(pcd_dir, os.path.relpath(las_fn, las_dir))
        tar_fn = os.path.splitext(tar_fn)[0] + '.npy'
        #debug_rgb(las_fn)
        if os.path.exists(tar_fn):
            print("Skipping " + las_fn + " : already processed")
        else:
            print("Converting " + las_fn + " ...")
            convert_single_file_xyz(las_fn, tar_fn)
        
    print("All done")

Main()