import os

mode = 1

cuDir = os.path.dirname(__file__)
os.chdir(cuDir)
if mode == 0:
        
    oldfile = r"../filenames/kitti_y.txt"
    newfile = r"../filenames/kitti_y1.txt"
    with open(oldfile, encoding="utf-8", mode='r') as oldfile1:
        with open(newfile, encoding="utf-8", mode='w') as newfile2:
            for i in oldfile1:
                a = i.rstrip().replace("\\", "/")
                if "_11.png" in a:
                    continue
                a1 = i.rstrip().replace(".png", ".pfm")
                b = a + " " + a.replace("image_2", "image_3") + " " + a.replace("image_2", "disp_occ_0") + " " + a1.replace("image_2", "kt15_full_disparity_plane/disp_occ_0/dense/dx") + " " + a1.replace("image_2", "kt15_full_disparity_plane/disp_occ_0/dense/dy")
                newfile2.write(b+'\n')

elif mode == 1:
    oldfile = r"../filenames/kitti_test_y.txt"
    newfile = r"../filenames/kitti_test_y1.txt"
    with open(oldfile, encoding="utf-8", mode='r') as oldfile1:
        with open(newfile, encoding="utf-8", mode='w') as newfile2:
            for i in oldfile1:
                a = i.rstrip().replace("\\", "/")
                if "_11.png" in a:
                    continue
                a1 = i.rstrip().replace(".png", ".pfm")
                b = a + " " + a.replace("image_2", "image_3")
                newfile2.write(b+'\n')
 