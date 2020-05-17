#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:19:00 2019

@author: agnus
"""

from datetime import datetime
import pandas as pd

def time_data(path, filename, metodo):
    f = open(path + filename, "r")
    FMT = '%Y-%m-%d %H:%M:%S,%f'
    
    ch = 0
    k = 1
    nf = []
    ft = []
    for x in f:
        #print x
        s = x[:23]
        #print s, len(s), len(s.strip())
        if "BOV" in metodo:
            if "le_descritores_train: starting" in x:
                startf = datetime.strptime(s, FMT)
            elif "bov_histogramas_grava test: ending(" in x:
                stopf = datetime.strptime(s, FMT)
                ft.append((stopf-startf).total_seconds())
                nf.append(k)
                k = k + 1
        elif "FV" in metodo:
            if "le_descritores_train: starting" in x:
                startf = datetime.strptime(s, FMT)
            elif "fv_fisher_vector for test:" in x:
                stopf = datetime.strptime(s, FMT)
                ft.append((stopf-startf).total_seconds())
                nf.append(k)
                k = k + 1
        else:
            if "Folder" in x and "ending" in x:
                nf.append(k)
                ft.append(float(x[x.find("(")+1: x.find(")")]))
                k = k + 1
        if ch==0:
            start = datetime.strptime(s, FMT)
            ch = 1
    
    f.close()
       
    stop = datetime.strptime(s, FMT)
            
    #print start, stop, stop-start, (stop-start).total_seconds()
    
    #d ={'folder': nf,'time': ft}
    d ={'time': ft}
    
    df = pd.DataFrame(data=d)
    
    
    return [s, (stop-start).total_seconds(), df.loc[:,"time"].mean(), df.loc[:,"time"].std()]


path_start = "/Projeto/Projetos/dat_artigo_"

metodos = ['_KAZE', 'A_KAZE_25', 'A_KAZE_50', 'AAKAZE_25', 'AFREAK', 'AKAZE', 'ASIFT', 'ASIFT_45', 'BOV__KAZE_25', 'BOV__KAZE_50', 'BOV__KAZE_100', 'BOV__KAZE_200', 'BOV__KAZE_250', 'BOV__KAZE_500', 'BOV__KAZE_1000', 'BOV_A_KAZE_25_25', 'BOV_A_KAZE_25_50', 'BOV_A_KAZE_25_100', 'BOV_A_KAZE_25_200', 'BOV_A_KAZE_25_250', 'BOV_A_KAZE_25_500', 'BOV_A_KAZE_25_1000', 'BOV_AAKAZE_25_25', 'BOV_AAKAZE_25_50', 'BOV_AAKAZE_25_100', 'BOV_AAKAZE_25_200', 'BOV_ABRIEF_35_50', 'BOV_ABRISK_35_50', 'BOV_AFREAK_35_50', 'BOV_AKAZE_25', 'BOV_AKAZE_50', 'BOV_AKAZE_100', 'BOV_AKAZE_200', 'BOV_AKAZE_250', 'BOV_AKAZE_500', 'BOV_AKAZE_1000', 'BOV_AORB_35_100', 'BOV_ASIFT_100_50', 'BOV_ASURF_35_50', 'BOV_ASURF_35_100', 'BOV_BRIEF_25', 'BOV_BRIEF_50', 'BOV_BRIEF_100', 'BOV_BRIEF_200', 'BOV_BRIEF_250', 'BOV_BRIEF_500', 'BOV_BRIEF_1000', 'BOV_BRISK_25', 'BOV_BRISK_50', 'BOV_BRISK_100', 'BOV_BRISK_200', 'BOV_BRISK_250', 'BOV_BRISK_500', 'BOV_BRISK_1000', 'BOV_FREAK_25', 'BOV_FREAK_50', 'BOV_FREAK_100', 'BOV_FREAK_200', 'BOV_FREAK_250', 'BOV_FREAK_500', 'BOV_FREAK_1000', 'BOV_ORB_25', 'BOV_ORB_50', 'BOV_ORB_100', 'BOV_ORB_200', 'BOV_ORB_250', 'BOV_ORB_500', 'BOV_ORB_1000', 'BOV_SIFT_25', 'BOV_SIFT_50', 'BOV_SIFT_100', 'BOV_SIFT_200', 'BOV_SIFT_250', 'BOV_SIFT_500', 'BOV_SIFT_1000', 'BOV_SURF_25', 'BOV_SURF_50', 'BOV_SURF_100', 'BOV_SURF_200', 'BOV_SURF_250', 'BOV_SURF_500', 'BOV_SURF_1000', 'BRIEF', 'BRISK', 'FREAK', 'FV__KAZE_2', 'FV__KAZE_3', 'FV__KAZE_4', 'FV__KAZE_5', 'FV__KAZE_10', 'FV__KAZE_15', 'FV_A_KAZE_25_2', 'FV_A_KAZE_25_3', 'FV_A_KAZE_25_4', 'FV_A_KAZE_25_5', 'FV_A_KAZE_25_10', 'FV_A_KAZE_50_2', 'FV_A_KAZE_50_3', 'FV_A_KAZE_50_4', 'FV_A_KAZE_50_5', 'FV_A_KAZE_50_10', 'FV_AAKAZE_25_2', 'FV_AAKAZE_25_3', 'FV_AAKAZE_25_4', 'FV_AAKAZE_25_5', 'FV_AAKAZE_25_10', 'FV_AKAZE_2', 'FV_AKAZE_3', 'FV_AKAZE_4', 'FV_AKAZE_5', 'FV_AKAZE_10', 'FV_AKAZE_15', 'FV_BRIEF_2', 'FV_BRIEF_3', 'FV_BRIEF_4', 'FV_BRIEF_5', 'FV_BRIEF_10', 'FV_BRIEF_15', 'FV_BRISK_2', 'FV_BRISK_3', 'FV_BRISK_4', 'FV_BRISK_5', 'FV_BRISK_10', 'FV_BRISK_15', 'FV_FREAK_2', 'FV_FREAK_3', 'FV_FREAK_4', 'FV_FREAK_5', 'FV_FREAK_10', 'FV_FREAK_15', 'FV_ORB_2', 'FV_ORB_3', 'FV_ORB_4', 'FV_ORB_5', 'FV_ORB_10', 'FV_ORB_15', 'FV_SIFT_2', 'FV_SIFT_3', 'FV_SIFT_4', 'FV_SIFT_5', 'FV_SIFT_10', 'FV_SIFT_15', 'FV_SURF_2', 'FV_SURF_3', 'FV_SURF_4', 'FV_SURF_5', 'FV_SURF_10', 'FV_SURF_15', 'ORB', 'SIFT', 'SURF'
           ]

seriais = ['201903271509', '201904091002', '201904091203', '201904092241', '201903151735', '201903271212', '201903150902', '201903181531', '201903271530', '201903271532', '201903271648', '201903271652', '201903271703', '201903272350', '201903272357', '201904082256', '201904082303', '201904082311', '201904082328', '201904082343', '201904090903', '201904090916', '201904101036', '201904101049', '201904101331', '201904101211', '201903221339', '201903221310', '201903221345', '201903271236', '201903271301', '201903271318', '201903271421', '201903271434', '201903271445', '201903271455', '201903221247', '201904051645', '201903221406', '201903222318', '201903191000', '201903181549', '201903181553', '201903181558', '201903251552', '201903251531', '201903251523', '201903181620', '201903181743', '201903181706', '201903181720', '201903251441', '201903251453', '201903251508', '201903181751', '201903181753', '201903181758', '201903181805', '201903251604', '201903251614', '201903251714', '201903181810', '201903181813', '201903181814', '201903181816', '201903251439', '201903251433', '201903251258', '201903252321', '201903252310', '201903252243', '201903252002', '201903251925', '201903251234', '201903251216', '201903181818', '201903190050', '201903190141', '201903190830', '201903251724', '201903251729', '201903251736', '201903131304', '201903131310', '201903131559', '201904051159', '201904051201', '201904051201', '201904051202', '201904051203', '201904051204', '201904082215', '201904082219', '201904082237', '201904082243', '201904082246', '201904091520', '201904091530', '201904091607', '201904091940', '201904091946', '201904092311', '201904101001', '201904101021', '201904101025', '201904101033', '201904051154', '201904051155', '201904051155', '201904051156', '201904051157', '201904051158', '201904051114', '201904051114', '201904051115', '201904051115', '201904051116', '201904051116', '201904051120', '201904051121', '201904051122', '201904051128', '201904051131', '201904051132', '201904051136', '201904051138', '201904051139', '201904051141', '201904051145', '201904051145', '201904051147', '201904051147', '201904051151', '201904051151', '201904051152', '201904051153', '201904051104', '201904051058', '201904051053', '201904051038', '201904051046', '201904051050', '201904051208', '201904051209', '201904051215', '201904051216', '201904051218', '201904051221', '201903131552', '201903131154', '201903131137'
           ]

ests = {}
for (metodo, serial) in zip(metodos,seriais):
    
    path =path_start+metodo+'/'
    filename = "sistema"+serial+".log"
    
    print path, metodo, filename
    
    l= time_data(path, filename, metodo)

    #print metodo
    #print "start time = ", l[0]
    #print "total time = ", l[1]
    #print "mean       = ", l[2]
    #print "std        = ", l[3]
    
    ests[metodo] = l

    
df = pd.DataFrame(data=ests,index=['start','total','fold mean','fold std']).T
print df.sort_values(['total'])
print df.to_csv(index=False)