import math

import numpy as np

def get_calibration_potenit():
    """Read calibration parameters from txt file:
    For the left color camera we use P2 which is K * [I|t]

    P = [fu, 0, x0, fu*t1-x0*t3
         0, fv, y0, fv*t2-y0*t3
         0, 0,  1,          t3]

    check also http://ksimek.github.io/2013/08/13/intrinsic/

    Simple case test:
    xyz = np.array([2, 3, 30, 1]).reshape(4, 1)
    xyz_2 = xyz[0:-1] + tt
    uv_temp = np.dot(kk, xyz_2)
    uv_1 = uv_temp / uv_temp[-1]
    kk_1 = np.linalg.inv(kk)
    xyz_temp2 = np.dot(kk_1, uv_1)
    xyz_new_2 = xyz_temp2 * xyz_2[2]
    xyz_fin_2 = xyz_new_2 - tt
    """
    kk =[ 
            [492.7495, 0., 311.4693],
            [0., 526.6584, 279.2586],
            [0., 0., 1.]
        ]   
    tt = [0., 0., 0.]

    kk_right = [
                  [493.4309, 0., 325.8401],
                  [0., 527.3013, 279.4993],
                  [0.,  0., 1.]
               ]
    tt_right = [-126.5094, -0.9646, 1.7356]

    kk = np.array(kk)
    tt = np.array(tt)

    kk_right = np.array(kk_right)
    tt_right = np.array(tt_right)
    
    kk, tt = get_translation(kk, tt)
    kk_right, tt_right = get_translation(kk_right, tt_right)

    return [kk, tt], [kk_right, tt_right]

def get_translation(kk, tt):
    """Separate intrinsic matrix from translation and convert in lists"""
    
    f_x = kk[0, 0]
    f_y = kk[1, 1]
    x0, y0 = kk[2, 0:2]
    aa, bb, t3 = tt
    t1 = float((aa - x0*t3) / f_x)
    t2 = float((bb - y0*t3) / f_y)
    tt = [t1, t2, float(t3)]
    return kk.tolist(), tt

def split_training_potenit(names_gt, path_train, path_val):
    """Split training and validation images"""
    # import pdb;pdb.set_trace()

    set_gt = set(names_gt)
    set_train = set()
    set_val = set()

    with open(path_train, "r") as f_train:
        for line in f_train:
            line = int(line.split('/')[-1].split('.')[0][1:])
            set_train.add(f'{line:06d}.txt')
    with open(path_val, "r") as f_val:
        for line in f_val:
            line = int(line.split('/')[-1].split('.')[0][1:])
            set_val.add(f'{line:06d}.txt')

    # import pdb;pdb.set_trace()

    set_train = tuple(set_gt.intersection(set_train))
    set_val = tuple(set_gt.intersection(set_val))
    assert set_train and set_val, "No validation or training annotations"
    return set_train, set_val