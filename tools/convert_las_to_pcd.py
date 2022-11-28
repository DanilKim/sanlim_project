import os
import glob
import laspy as lp

def convert_single_file_xyzrgb(las_fn, pcd_fn):

    cloud = lp.read(las_fn)
    
    length = len(cloud)
    print('Total lines of {}: '.format(las_fn), length)
    
    pcdFile = open(pcd_fn, "w") # pcd-file

    pcdFile.write("VERSION .7\nFIELDS x y z r g b\nSIZE 4 4 4 1 1 1\nTYPE F F F U U U\nCOUNT 1 1 1 1 1 1\nWIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA ascii\n".format(length, length)) # Sets the header of pcd in a specific format, see more on http://pointclouds.org/documentation/tutorials/pcd_file_format.php
    #pcdFile.write("VERSION .7\nFIELDS x y z \nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA ascii\n".format(length, length)) # Sets the header of pcd in a specific format, see more on http://pointclouds.org/documentation/tutorials/pcd_file_format.php


    for i, (x, y, z, r, g, b) in enumerate(zip(cloud.x, cloud.y, cloud.z, cloud.red, cloud.blue, cloud.green)):
        #if(i % 50000 == 0):
        #    print("Current file position: " + str(i))

        pcdFile.write(" ".join([
            str(x), # x-value
            str(y), # y-value
            str(z), # z-value
            str(int(r>>8)),
            str(int(g>>8)),
            str(int(b>>8)),
            #"{:e}".format((int(r)<<16) + (int(g)<<8) + int(b)) # rgb value renderd in scientific format
        ]) + "\n")
            
    pcdFile.close()


def convert_single_file_xyz(las_fn, pcd_fn):

    cloud = lp.read(las_fn)
    
    length = len(cloud)
    print('Total lines of {}: '.format(las_fn), length)
    
    pcdFile = open(pcd_fn, "w") # pcd-file

    pcdFile.write("VERSION .7\nFIELDS x y z r g b\nSIZE 4 4 4 1 1 1\nTYPE F F F U U U\nCOUNT 1 1 1 1 1 1\nWIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA ascii\n".format(length, length)) # Sets the header of pcd in a specific format, see more on http://pointclouds.org/documentation/tutorials/pcd_file_format.php
    #pcdFile.write("VERSION .7\nFIELDS x y z \nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA ascii\n".format(length, length)) # Sets the header of pcd in a specific format, see more on http://pointclouds.org/documentation/tutorials/pcd_file_format.php

    for i, (x, y, z, r, g, b) in enumerate(zip(cloud.x, cloud.y, cloud.z, cloud.red, cloud.blue, cloud.green)):
        #if(i % 50000 == 0):
        #    print("Current file position: " + str(i))

        pcdFile.write(" ".join([
            str(x), # x-value
            str(y), # y-value
            str(z), # z-value
            str(int(r>>8)),
            str(int(g>>8)),
            str(int(b>>8)),
            #"{:e}".format((int(r)<<16) + (int(g)<<8) + int(b)) # rgb value renderd in scientific format
        ]) + "\n")
            
    pcdFile.close()
    
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
    
    pcd_dir = os.path.join('sanlim_pcd_rgb')
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
            convert_single_file_xyzrgb(las_fn, tar_fn)
        
    print("All done")

Main()