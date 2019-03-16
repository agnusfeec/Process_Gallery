#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 19:40:57 2018

@author: agnus
"""
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool

kp_detector = None
ds_extractor = None

# local modules
#from common import Timer
#from find_obj import init_feature, filter_matches, explore_match


#%%
def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    Ai = cv.invertAffineTransform(A)

    #img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
    #plt.imshow(img, cmap='gray'),plt.show()

    return img, mask, Ai

#%%

#https://stackoverflow.com/questions/10274774/python-elegant-and-efficient-ways-to-mask-a-list
from itertools import compress
class MaskableList(list):
    def __getitem__(self, index):
        try: return super(MaskableList, self).__getitem__(index)
        except TypeError: return MaskableList(compress(self, index))

#%%
def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''

    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)

        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        #keypoints, descrs = process(detector, timg, tmask)

        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
            return keypoints, descrs
        else:
            if not(t==1.0 and phi==0.0):
                mask = np.random.choice([False, True], len(descrs), p=[0.75, 0.25])
            else:
                mask = np.random.choice([False, True], len(descrs), p=[0.0, 1.0])

            mlist = MaskableList
            keyp = mlist(keypoints)[mask]
            return keyp, descrs[mask]

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()
    return keypoints, np.array(descrs)

#%%
def ds_read(sift_folder, filename, sift_type):
    import csv
    import numpy as np
    import os

    ds_type = ''
    ds_num = 0
    ds_dim = 0

    fname = os.path.join(sift_folder, filename[:-5] + '_' + sift_type + '_des.csv')

    with open(fname, 'rb') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        rownumber = 0

        aux = []

        for row in spamreader:

            if rownumber == 0:

                ds_type = row[0]
                ds_num = row[1]
                ds_dim = row[2]
                rownumber = rownumber + 1

            else:

                aux.append(row)

        ds = np.asarray(aux, dtype=np.float)

        return ds

#%%
def kp_read(sift_folder, filename, sift_type):

    import csv
    import numpy as np
    import os

    kp_type = ''
    kp_num = 0

    fname = os.path.join(sift_folder, filename[:-5] + '_' + sift_type + '_kp.csv')

    with open(fname, 'rb') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        rownumber = 0

        aux = []

        for row in spamreader:

            if rownumber == 0:

                kp_type = row[0]
                kp_num = row[1]
                rownumber = rownumber + 1

            else:

                aux.append(row)

        kp = np.asarray(aux, dtype=np.float32)

        return kp, kp_type, kp_num

#%%
#https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
def kp_write(filename, tipo, kp):
    import csv

    file_path = '/media/sf_Projeto/Projetos/dataset/descriptors'

    if not (kp is None) :
        with open(file_path + filename, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([tipo, len(kp)])
            for point in kp:
                spamwriter.writerow([point.pt[0], point.pt[1], point.size, point.angle, point.response, point.octave, point.class_id])


#%%
def des_write(filename, tipo, des):
    import csv

    file_path = '/media/sf_Projeto/Projetos/dataset/descriptors'

    if not(des is None) :
        with open(file_path + filename, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([tipo, len(des), len(des[0])])
            for desc in des:
                spamwriter.writerow(desc)


#%%
def fast_do(relevant_path, files):

    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt

    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()

    for img_file in files:

        img_filename = relevant_path + "/" + img_file[:-1]

        img = cv.imread(img_filename,0)

        #Histograms Equalization in OpenCV
        #img = cv.equalizeHist(aux)

        #CLAHE (Contrast Limited Adaptive Histogram Equalization)
        #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #img = clahe.apply(aux)

        # find and draw the keypoints
        kp = fast.detect(img,None)
        img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
        plt.imshow(img2),plt.show()
        # Print all default params
        #print( "Threshold: {}".format(fast.getThreshold()) )
        #print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
        #print( "neighborhood: {}".format(fast.getType()) )
        print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

        kp_write(img_file[7:-5]+'_fast_kp.csv','fast', kp)
        #des_write(img_file[7:-5]+'_fast_des.csv','fast', des)

        #cv.imwrite('fast_true.png',img2)

        # Disable nonmaxSuppression
        #fast.setNonmaxSuppression(0)
        #kp = fast.detect(img,None)

        #print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )

        #img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

        #cv.imwrite('fast_false.png',img3)


#%%
def sift_do(relevant_path, files):

    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt

    sift = cv.xfeatures2d.SIFT_create()

    for img_file in files:

        img_filename = relevant_path + "/" + img_file[:-1]

        aux = cv.imread(img_filename,0)

        #Histograms Equalization in OpenCV
        #img = cv.equalizeHist(aux)

        #CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(aux)

        # Find keypoints and descriptors directly
        kp, des = sift.detectAndCompute(img,None)
        print( len(kp) )

        kp_write(img_file[7:-5]+'_sift_kp.csv','sift', kp)
        des_write(img_file[7:-5]+'_sift_des.csv','sift', des)

        #img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
        #plt.imshow(img2),plt.show()


#%%
def surf_do(relevant_path, files):

    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt

    # Create SURF object. You can specify params here or later.
    # Here I set Hessian Threshold to 400
    surf = cv.xfeatures2d.SURF_create(300)


    for img_file in files:

        img_filename = relevant_path + "/" + img_file[:-1]

        aux = cv.imread(img_filename,0)

        #Histograms Equalization in OpenCV
        #img = cv.equalizeHist(aux)

        #CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(aux)

        # Find keypoints and descriptors directly
        #kp, des = surf.detectAndCompute(img,None)
        #print( len(kp) )

        # Check present Hessian threshold
        #print( surf.getHessianThreshold() )

        # We set it to some 50000. Remember, it is just for representing in picture.
        # In actual cases, it is better to have a value 300-500
        surf.setHessianThreshold(300)

        # Again compute keypoints and check its number.
        kp, des = surf.detectAndCompute(img,None)
        print( len(kp) )
        print( surf.getHessianThreshold() )

        kp_write(img_file[7:-5]+'_surf_kp.csv','surf', kp)
        des_write(img_file[7:-5]+'_surf_des.csv','surf', des)

        #img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
        #plt.imshow(img2),plt.show()


#%%
def brief_do(relevant_path, files):

    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt

    # Initiate FAST detector
    star = cv.xfeatures2d.StarDetector_create(20,15)

    # Initiate BRIEF extractor
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    for img_file in files:

        img_filename = relevant_path + "/" + img_file[:-1]

        aux = cv.imread(img_filename,0)

        #Histograms Equalization in OpenCV
        #img = cv.equalizeHist(aux)

        #CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img1 = clahe.apply(aux)

        #kernel = np.ones((8,8),np.float32)/64
        #img = cv.filter2D(aux,-1,kernel)

        #img = cv.GaussianBlur(img1,(7,7),0)
        #img = aux
        # find the keypoints with STAR
        kp = star.detect(img1,None)
        # compute the descriptors with BRIEF
        kp, des = brief.compute(img1, kp)

        #if len(kp)==0:
        #    print img_file[:-1]

        print( len(kp), img_file[:-1] )

        kp_write(img_file[7:-5]+'_brief_kp.csv','brief', kp)
        des_write(img_file[7:-5]+'_brief_des.csv','brief', des)

        #img2 = cv.drawKeypoints(img1,kp,None,(255,0,0),4)
        #plt.imshow(img2),plt.show()


#%%
def orb_do(relevant_path, files):

    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt

    # Initiate ORB detector
    orb = cv.ORB_create(500,1.8,8,20)

    for img_file in files:

        img_filename = relevant_path + "/" + img_file[:-1]

        aux = cv.imread(img_filename,0)

        #Histograms Equalization in OpenCV
        #img = cv.equalizeHist(aux)

        #CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(aux)

        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)

        print( len(kp) )

        kp_write(img_file[7:-5]+'_orb_kp.csv','orb', kp)
        des_write(img_file[7:-5]+'_orb_des.csv','orb', des)

        #img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
        #plt.imshow(img2),plt.show()


#%%
def brisk_do(relevant_path, files):

    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt

    # Initiate ORB detector
    brisk = cv.BRISK_create()

    for img_file in files:

        img_filename = relevant_path + "/" + img_file[:-1]

        aux = cv.imread(img_filename,0)

        #Histograms Equalization in OpenCV
        #img = cv.equalizeHist(aux)

        #CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(aux)

        # find the keypoints with ORB
        kp = brisk.detect(img,None)
        # compute the descriptors with ORB
        kp, des = brisk.compute(img, kp)

        print( len(kp) )

        kp_write(img_file[7:-5]+'_brisk_kp.csv','brisk', kp)
        des_write(img_file[7:-5]+'_brisk_des.csv','brisk', des)

        #img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
        #plt.imshow(img2),plt.show()



#%%
def freak_do(relevant_path, files):

    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt

    # Initiate FAST detector
    star = cv.xfeatures2d.StarDetector_create(20,15)

    # Initiate FREAK extractor
    freak = cv.xfeatures2d.FREAK_create()

    for img_file in files:

        img_filename = relevant_path + "/" + img_file[:-1]

        aux = cv.imread(img_filename,0)

        #Histograms Equalization in OpenCV
        #img = cv.equalizeHist(aux)

        #CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(aux)

        # find the keypoints with STAR
        kp = star.detect(img,None)

        # find the keypoints with FREAK
        kp, des = freak.compute(img, kp)

        print( len(kp) )

        kp_write(img_file[7:-5]+'_freak_kp.csv','freak', kp)
        des_write(img_file[7:-5]+'_freak_des.csv','freak', des)

        #img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
        #plt.imshow(img2),plt.show()


#%%
class myDetector:

    def __init__(self, type='sift'):

        self.type = type
        #self.matcher_type = matcher_type

        if type == 'sift':
            self.extractor = cv.xfeatures2d.SIFT_create()
        elif type == 'surf':
            self.extractor = cv.xfeatures2d.SURF_create(800)
        elif type == 'orb':
            self.extractor = cv.ORB_create(400)
        elif type == 'akaze':
            self.extractor = cv.AKAZE_create()
        elif type == 'brisk':
            self.extractor = cv.BRISK_create()
        elif type == 'freak':
            self.detector = cv.xfeatures2d.StarDetector_create(20,15)
            self.extractor = cv.xfeatures2d.FREAK_create()
            self.norm = cv.NORM_HAMMING
        else:
            self.detector = None
            self.extractor = None
            self.norm = None

    def detectAndCompute(self, timg, tmask):
        if self.detector is None:
            kp, ds = self.extractor.detectAndCompute(timg,tmask)
        else:
            kp = self.detector.detect(timg, tmask)
            kp, ds = self.extractor.compute(timg, kp)

        return kp, ds

def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv.xfeatures2d.SIFT_create()
        norm = cv.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv.xfeatures2d.SURF_create(800)
        norm = cv.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv.ORB_create(400)
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'freak':
        detector = cv.FREAK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv.BFMatcher(norm)

    return detector, matcher

#%%
def asift_do(relevant_path, files, feature_name = "sift"):

    import cv2 as cv
    import numpy as np
    from matplotlib import pyplot as plt
    #from find_obj import init_feature

    #detector = cv.xfeatures2d.SIFT_create()
    #feature_name = "freak"

    #detector, matcher = init_feature(feature_name)

    detector = myDetector(feature_name)

    for img_file in files:

        img_filename = relevant_path + "/" + img_file[:-1]

        aux = cv.imread(img_filename,0)

        #Histograms Equalization in OpenCV
        #img = cv.equalizeHist(aux)

        #CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(aux)
        #img = aux

        pool=ThreadPool(processes = 4) #cv.getNumberOfCPUs())
        kp, des = affine_detect(detector, img, pool=pool)

        print( len(kp) )

        kp_write(img_file[6:-5]+'_a' + feature_name + '_kp.csv','a' + feature_name + '', kp)
        des_write(img_file[6:-5]+'_a' + feature_name + '_des.csv','a' + feature_name + '', des)

        #img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
        #plt.imshow(img2),plt.show()

#relevant_path = "/home/agnus/Documentos/dataset/tatt-c/tattoo_identification/test"
#relevant_path = "/media/agnus/My Passport/Projeto/dataset/tatt-c/tattoo_identification/training"
#gallery_file = "group.txt"

#relevant_path = "/media/agnus/My Passport/Projeto/dataset/tatt-c/background/images/"
#gallery_file = "bg.txt"

#relevant_path = "/media/agnus/My Passport/Projeto/dataset/tatt-c/tattoo_identification/test"
#gallery_file = "gallery_small.txt"
#gallery_file = "probes.txt"

relevant_path = "/media/sf_Projeto/dataset/tatt-c/tattoo_identification/test/"
#gallery_file = "probes.txt"
gallery_file = "gallery_small.txt"

#relevant_path = "/media/sf_Projeto/dataset/tatt-c/tattoo_identification/training"
#gallery_file = "group.txt"

#relevant_path = "/media/sf_Projeto/dataset/tatt-c/background/images/"
#gallery_file = "cropped.txt"

# Open the file with read only permit
f = open(relevant_path + "/"+ gallery_file, "r")
# use readlines to read all lines in the file
# The variable "lines" is a list containing all lines in the file
# read all lines at once
lines = list(f)
# close the file after reading the lines.
f.close()

#print(lines)

#surf_do(relevant_path, lines)
#brief_do(relevant_path, lines)
#fast_do(relevant_path, lines)
#orb_do(relevant_path, lines)
#sift_do(relevant_path, lines)
#brisk_do(relevant_path, lines)
#freak_do(relevant_path, lines)
asift_do(relevant_path, lines, 'freak')

#files = lines

#for img_file in files:
#    print img_file
#    #kp_read(img_file[7:-5]+'_brief_kp.csv','orb', kp)
#    kp, _, _ = kp_read('', img_file[7:], 'surf')
#    break
