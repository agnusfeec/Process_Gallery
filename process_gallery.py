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
#from matplotlib import pyplot as plt

# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool

#from lib_files import *
import lib_files as lbf

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
def affine_detect(detector, consider, img, mask=None, pool=None ):
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

    def f(param):
        t, phi = param
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
                mask = np.random.choice([False, True], len(descrs), p=[1.0-consider/100.0, consider/100.0])
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
        #print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    #print()
    return keypoints, np.array(descrs)

#%%
def fast_do(relevant_path, files):

    #import numpy as np
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

        lbf.kp_write(img_file[7:-5]+'_fast_kp.csv','fast', kp)
        #des_write(img_file[7:-5]+'_fast_des.csv','fast', des)

        #cv.imwrite('fast_true.png',img2)

        # Disable nonmaxSuppression
        #fast.setNonmaxSuppression(0)
        #kp = fast.detect(img,None)

        #print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )

        #img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

        #cv.imwrite('fast_false.png',img3)


class myDetector:

    def __init__(self, type='sift'):

        self.type = type
        #self.matcher_type = matcher_type

        if type == 'sift':
            self.detector = cv.xfeatures2d.SIFT_create()
            self.extractor = None
        elif type == 'surf':
            self.detector = cv.xfeatures2d.SURF_create(300)
            self.extractor = None
        elif type == 'orb':
            self.detector = cv.ORB_create(400)
            self.extractor = None
        elif type == 'akaze':
            self.detector = cv.AKAZE_create()
            self.extractor = None
        elif type == '_kaze':
            self.detector = cv.KAZE_create()
            self.extractor = None
        elif type == 'brisk':
            self.detector = cv.BRISK_create()
            self.extractor = None
        elif type == 'freak':
            self.detector = cv.xfeatures2d.StarDetector_create(20,15)
            self.extractor = cv.xfeatures2d.FREAK_create()
            self.norm = cv.NORM_HAMMING
        elif type == 'brief':
            self.detector = cv.xfeatures2d.StarDetector_create(20,15)
            self.extractor = cv.xfeatures2d.BriefDescriptorExtractor_create()
        else:
            self.extractor = cv.xfeatures2d.FREAK_create()
            self.detector = None
            self.extractor = None
            self.norm = None

    def detectAndCompute(self, timg, tmask):
        if self.extractor is None:
            kp, ds = self.detector.detectAndCompute(timg,tmask)
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
            flann_params = dict(algorithm = cv.FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = cv.FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv.BFMatcher(norm)

    return detector, matcher

#%%
def descriptor_do(relevant_path, file_path, files, feature_name = "sift"):

    import os
    import cv2 as cv
    from math import sqrt
    #import numpy as np
    #from matplotlib import pyplot as plt
    
    detector = myDetector(feature_name)

    for img_file in files:

        img_filename = relevant_path + "/" + img_file[:-1]
        
        
        pp = img_file.find("/")
        filename1 = img_file[pp:-5]+'_' + feature_name + '_kp.csv' 
        filename2 = img_file[pp:-5]+'_' + feature_name + '_des.csv'
        
        if not (os.path.isfile(file_path + filename1) and os.path.isfile(file_path + filename2)) :
     
            aux = cv.imread(img_filename,0)
    
            #Histograms Equalization in OpenCV
            #img = cv.equalizeHist(aux)
    
            #CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            img1 = clahe.apply(aux)
    
            # compute the descriptors with BRIEF
            kp, des = detector.detectAndCompute(img1, None)
    
            #print( len(kp), img_file[:-1] )
            
            if len(kp)<10:
                
                k = sqrt((240.0*480.0*1.0)/(img1.shape[0]*img1.shape[1]))
                img2 = cv.resize(img1,None,fx=k, fy=k, interpolation = cv.INTER_CUBIC)
    
                # compute the descriptors with BRIEF
                kp, des = detector.detectAndCompute(img2, None)
        
                if len(kp) == 0:
                    print( len(kp), img_file[:-1] )
    
            pp = img_file.find("/")
            #print( file_path, img_file[pp:-5], img_file[pp:-5]+'_' + feature_name + '_kp.csv', feature_name )
            lbf.kp_write(file_path, img_file[pp:-5]+'_' + feature_name + '_kp.csv', feature_name, kp)
            lbf.des_write(file_path, img_file[pp:-5]+'_' + feature_name + '_des.csv', feature_name, des)
    
            #img2 = cv.drawKeypoints(img1,kp,None,(255,0,0),4)
            #plt.imshow(img2),plt.show()

        print (img_file[:-1])

#%%
def asift_do(relevant_path, file_path, files, feature_name, consider):

    import os
    import cv2 as cv
    #import numpy as np
    #from matplotlib import pyplot as plt
    #from find_obj import init_feature

    #detector = cv.xfeatures2d.SIFT_create()
    #feature_name = "freak"

    #detector, matcher = init_feature(feature_name)

    detector = myDetector(feature_name)

    for img_file in files:

        img_filename = relevant_path + "/" + img_file[:-1]

        pp = img_file.find("/")
        filename1 = img_file[pp:-5]+'_a' + feature_name + '_' + str(consider).zfill(3) + '_kp.csv' 
        filename2 = img_file[pp:-5]+'_a' + feature_name + '_' + str(consider).zfill(3) + '_des.csv'
        
        if not (os.path.isfile(file_path + filename1) and os.path.isfile(file_path + filename2)) :
        
            aux = cv.imread(img_filename,0)
            (w,h) = aux.shape
            if(w*h)<10000:
                res = cv.resize(aux,None,fx=1.5, fy=1.5, interpolation = cv.INTER_CUBIC)
                aux = res
                
            #Histograms Equalization in OpenCV
            #img = cv.equalizeHist(aux)
    
            #CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            img = clahe.apply(aux)
            #img = aux
    
            pool=ThreadPool(processes = 1) #cv.getNumberOfCPUs()-1) #check the case where CPU=1
            kp, des = affine_detect(detector, consider, img, pool=pool)
    
            #print (img_file[:-1], len(kp), img_file[6:-5]+'_a' + feature_name + '_' + str(consider) + '_kp.csv')
            
            #if(len(kp)>0):
    
            #    lbf.kp_write(file_path, img_file[6:-5]+'_a' + feature_name + '_' + str(consider) + '_kp.csv','a' + feature_name + '', kp)
            #    lbf.des_write(file_path, img_file[6:-5]+'_a' + feature_name + '_' + str(consider) + '_des.csv','a' + feature_name + '', des)
    
            if len(kp) == 0:
                print( len(kp), img_file[:-1] )
    
            pp = img_file.find("/")
            #print( file_path, img_file[pp:-5], img_file[pp:-5]+'_a' + feature_name + '_kp.csv', 'a' + feature_name )
            lbf.kp_write(file_path, filename1, 'a' + feature_name, kp)
            lbf.des_write(file_path, filename2, 'a' + feature_name, des)
    
            #img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
            #plt.imshow(img2),plt.show()

        
        print (img_file[:-1])
            
#%%
        
file_path = '/Projeto/dataset/descriptors'


r_path1 = "/Projeto/dataset/tatt-c/tattoo_identification/test"
g_file11 = "probes.txt"
g_file12 = "gallery_small.txt"

r_path2 = "/Projeto/dataset/tatt-c/tattoo_identification/training"
g_file21 = "group.txt"

r_path3 = '/Projeto/dataset/tatt-c/background/images'
g_file31 = 'cropped.txt'
#sub_sets = [(r_path1, g_file1),(r_path1, g_file2),(r_path2, g_file3)]
sub_sets = [(r_path1, g_file11),(r_path1, g_file12),(r_path2, g_file21),(r_path3, g_file31)] #,
#sub_sets = [(r_path3, g_file31)]


#names = ['_kaze', 'akaze', 'freak', 'brief', 'brisk', 'orb', 'surf', 'sift']
#names = ['akaze', 'freak', 'brief', 'brisk', 'orb', 'sift', 'surf']
#names = ['orb', 'sift', 'surf']
names = ['sift']

for (relevant_path, gallery_file) in sub_sets:
    
    print (relevant_path, gallery_file)
    f = open(relevant_path + "/"+ gallery_file, "r")
    lines = list(f)
    f.close()

    print (names)
    
    for name in names:
        print(name)
        #asift_do(relevant_path, file_path, lines, name, 50)
        descriptor_do(relevant_path, file_path, lines, name)
