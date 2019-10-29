#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:37:13 2019

@author: agnus
"""

import lib_sistema_02 as ls
import os

path_start = "/Projeto/Projetos/"

metodos = ['dat_artigo_AAKAZE_50', 'dat_artigo_ABRIEF_50', 
           'dat_artigo_ABRISK_50', 'dat_artigo_AFREAK_50', 'dat_artigo_AKAZE', 
           'dat_artigo_A_KAZE_50', 'dat_artigo_AORB_50', 'dat_artigo_ASIFT_50',
           'dat_artigo_ASURF_50', 'dat_artigo_BOV_AAKAZE_50_00010', 
           'dat_artigo_BOV_AAKAZE_50_00025', 'dat_artigo_BOV_AAKAZE_50_00050', 
           'dat_artigo_BOV_AAKAZE_50_00100', 'dat_artigo_BOV_AAKAZE_50_00200', 
           'dat_artigo_BOV_AAKAZE_50_00500', 'dat_artigo_BOV_AAKAZE_50_01000', 
           'dat_artigo_BOV_ABRIEF_50_00010', 'dat_artigo_BOV_ABRIEF_50_00025', 
           'dat_artigo_BOV_ABRIEF_50_00050', 'dat_artigo_BOV_ABRIEF_50_00100', 
           'dat_artigo_BOV_ABRIEF_50_00200', 'dat_artigo_BOV_ABRIEF_50_00500', 
           'dat_artigo_BOV_ABRIEF_50_01000', 'dat_artigo_BOV_ABRISK_50_00010', 
           'dat_artigo_BOV_ABRISK_50_00025', 'dat_artigo_BOV_ABRISK_50_00050', 
           'dat_artigo_BOV_ABRISK_50_00100', 'dat_artigo_BOV_ABRISK_50_00200', 
           'dat_artigo_BOV_ABRISK_50_00500', 'dat_artigo_BOV_ABRISK_50_01000', 
           'dat_artigo_BOV_AFREAK_50_00010', 'dat_artigo_BOV_AFREAK_50_00025', 
           'dat_artigo_BOV_AFREAK_50_00050', 'dat_artigo_BOV_AFREAK_50_00100', 
           'dat_artigo_BOV_AFREAK_50_00200', 'dat_artigo_BOV_AFREAK_50_00500', 
           'dat_artigo_BOV_AFREAK_50_01000', 'dat_artigo_BOV_AKAZE_00010', 
           'dat_artigo_BOV_AKAZE_00025', 'dat_artigo_BOV_AKAZE_00050', 
           'dat_artigo_BOV_AKAZE_00100', 'dat_artigo_BOV_AKAZE_00200', 
           'dat_artigo_BOV_AKAZE_00500', 'dat_artigo_BOV_AKAZE_01000', 
           'dat_artigo_BOV_A_KAZE_50_00010', 'dat_artigo_BOV_A_KAZE_50_00025', 
           'dat_artigo_BOV_A_KAZE_50_00050', 'dat_artigo_BOV_A_KAZE_50_00100', 
           'dat_artigo_BOV_A_KAZE_50_00200', 'dat_artigo_BOV_A_KAZE_50_00500', 
           'dat_artigo_BOV_A_KAZE_50_01000', 'dat_artigo_BOV_AORB_50_00010', 
           'dat_artigo_BOV_AORB_50_00025', 'dat_artigo_BOV_AORB_50_00050', 
           'dat_artigo_BOV_AORB_50_00100', 'dat_artigo_BOV_AORB_50_00200', 
           'dat_artigo_BOV_AORB_50_00500', 'dat_artigo_BOV_AORB_50_01000', 
           'dat_artigo_BOV_ASIFT_50_00010', 'dat_artigo_BOV_ASIFT_50_00025', 
           'dat_artigo_BOV_ASIFT_50_00050', 'dat_artigo_BOV_ASIFT_50_00100', 
           'dat_artigo_BOV_ASIFT_50_00200', 'dat_artigo_BOV_ASIFT_50_00500', 
           'dat_artigo_BOV_ASIFT_50_01000', 'dat_artigo_BOV_ASURF_50_00010', 
           'dat_artigo_BOV_ASURF_50_00025', 'dat_artigo_BOV_ASURF_50_00050', 
           'dat_artigo_BOV_ASURF_50_00100', 'dat_artigo_BOV_ASURF_50_00200', 
           'dat_artigo_BOV_ASURF_50_00500', 'dat_artigo_BOV_ASURF_50_01000', 
           'dat_artigo_BOV_BRIEF_00010', 'dat_artigo_BOV_BRIEF_00025', 
           'dat_artigo_BOV_BRIEF_00050', 'dat_artigo_BOV_BRIEF_00100', 
           'dat_artigo_BOV_BRIEF_00200', 'dat_artigo_BOV_BRIEF_00500', 
           'dat_artigo_BOV_BRIEF_01000', 'dat_artigo_BOV_BRISK_00010', 
           'dat_artigo_BOV_BRISK_00025', 'dat_artigo_BOV_BRISK_00050', 
           'dat_artigo_BOV_BRISK_00100', 'dat_artigo_BOV_BRISK_00200', 
           'dat_artigo_BOV_BRISK_00500', 'dat_artigo_BOV_BRISK_01000', 
           'dat_artigo_BOV_FREAK_00010', 'dat_artigo_BOV_FREAK_00025', 
           'dat_artigo_BOV_FREAK_00050', 'dat_artigo_BOV_FREAK_00100', 
           'dat_artigo_BOV_FREAK_00200', 'dat_artigo_BOV_FREAK_00500', 
           'dat_artigo_BOV_FREAK_01000', 'dat_artigo_BOV__KAZE_00010', 
           'dat_artigo_BOV__KAZE_00025', 'dat_artigo_BOV__KAZE_00050', 
           'dat_artigo_BOV__KAZE_00100', 'dat_artigo_BOV__KAZE_00200', 
           'dat_artigo_BOV__KAZE_00500', 'dat_artigo_BOV__KAZE_01000', 
           'dat_artigo_BOV_ORB_00010', 'dat_artigo_BOV_ORB_00025', 
           'dat_artigo_BOV_ORB_00050', 'dat_artigo_BOV_ORB_00100', 
           'dat_artigo_BOV_ORB_00200', 'dat_artigo_BOV_ORB_00500', 
           'dat_artigo_BOV_ORB_01000', 'dat_artigo_BOV_SIFT_00010', 
           'dat_artigo_BOV_SIFT_00025', 'dat_artigo_BOV_SIFT_00050', 
           'dat_artigo_BOV_SIFT_00100', 'dat_artigo_BOV_SIFT_00200', 
           'dat_artigo_BOV_SIFT_00500', 'dat_artigo_BOV_SIFT_01000', 
           'dat_artigo_BOV_SURF_00010', 'dat_artigo_BOV_SURF_00025', 
           'dat_artigo_BOV_SURF_00050', 'dat_artigo_BOV_SURF_00100', 
           'dat_artigo_BOV_SURF_00200', 'dat_artigo_BOV_SURF_00500', 
           'dat_artigo_BOV_SURF_01000', 'dat_artigo_BRIEF', 'dat_artigo_BRISK', 
           'dat_artigo_FREAK', 'dat_artigo_FV_AAKAZE_50_00001', 
           'dat_artigo_FV_AAKAZE_50_00002', 'dat_artigo_FV_AAKAZE_50_00003', 
           'dat_artigo_FV_AAKAZE_50_00004', 'dat_artigo_FV_AAKAZE_50_00005', 
           'dat_artigo_FV_AAKAZE_50_00010', 'dat_artigo_FV_AAKAZE_50_00015', 
           'dat_artigo_FV_ABRIEF_50_00001', 'dat_artigo_FV_ABRIEF_50_00002', 
           'dat_artigo_FV_ABRIEF_50_00003', 'dat_artigo_FV_ABRIEF_50_00004', 
           'dat_artigo_FV_ABRIEF_50_00005', 'dat_artigo_FV_ABRIEF_50_00010', 
           'dat_artigo_FV_ABRIEF_50_00015', 'dat_artigo_FV_ABRISK_50_00001', 
           'dat_artigo_FV_ABRISK_50_00002', 'dat_artigo_FV_ABRISK_50_00003', 
           'dat_artigo_FV_ABRISK_50_00004', 'dat_artigo_FV_ABRISK_50_00005', 
           'dat_artigo_FV_ABRISK_50_00010', 'dat_artigo_FV_ABRISK_50_00015', 
           'dat_artigo_FV_AFREAK_50_00001', 'dat_artigo_FV_AFREAK_50_00002', 
           'dat_artigo_FV_AFREAK_50_00003', 'dat_artigo_FV_AFREAK_50_00004', 
           'dat_artigo_FV_AFREAK_50_00005', 'dat_artigo_FV_AFREAK_50_00010', 
           'dat_artigo_FV_AFREAK_50_00015', 'dat_artigo_FV_AKAZE_00001', 
           'dat_artigo_FV_AKAZE_00002', 'dat_artigo_FV_AKAZE_00003', 
           'dat_artigo_FV_AKAZE_00004', 'dat_artigo_FV_AKAZE_00005', 
           'dat_artigo_FV_AKAZE_00010', 'dat_artigo_FV_AKAZE_00015', 
           'dat_artigo_FV_A_KAZE_50_00001', 'dat_artigo_FV_A_KAZE_50_00002', 
           'dat_artigo_FV_A_KAZE_50_00003', 'dat_artigo_FV_A_KAZE_50_00004', 
           'dat_artigo_FV_A_KAZE_50_00005', 'dat_artigo_FV_A_KAZE_50_00010', 
           'dat_artigo_FV_A_KAZE_50_00015', 'dat_artigo_FV_AORB_50_00001', 
           'dat_artigo_FV_AORB_50_00002', 'dat_artigo_FV_AORB_50_00003', 
           'dat_artigo_FV_AORB_50_00004', 'dat_artigo_FV_AORB_50_00005', 
           'dat_artigo_FV_AORB_50_00010', 'dat_artigo_FV_AORB_50_00015', 
           'dat_artigo_FV_ASIFT_50_00001', 'dat_artigo_FV_ASIFT_50_00002', 
           'dat_artigo_FV_ASIFT_50_00003', 'dat_artigo_FV_ASIFT_50_00004', 
           'dat_artigo_FV_ASIFT_50_00005', 'dat_artigo_FV_ASIFT_50_00010', 
           'dat_artigo_FV_ASIFT_50_00015', 'dat_artigo_FV_ASURF_50_00001', 
           'dat_artigo_FV_ASURF_50_00002', 'dat_artigo_FV_ASURF_50_00003', 
           'dat_artigo_FV_ASURF_50_00004', 'dat_artigo_FV_ASURF_50_00005', 
           'dat_artigo_FV_ASURF_50_00010', 'dat_artigo_FV_ASURF_50_00015', 
           'dat_artigo_FV_BRIEF_00001', 'dat_artigo_FV_BRIEF_00002', 
           'dat_artigo_FV_BRIEF_00003', 'dat_artigo_FV_BRIEF_00004', 
           'dat_artigo_FV_BRIEF_00005', 'dat_artigo_FV_BRIEF_00010', 
           'dat_artigo_FV_BRIEF_00015', 'dat_artigo_FV_BRISK_00001', 
           'dat_artigo_FV_BRISK_00002', 'dat_artigo_FV_BRISK_00003', 
           'dat_artigo_FV_BRISK_00004', 'dat_artigo_FV_BRISK_00005', 
           'dat_artigo_FV_BRISK_00010', 'dat_artigo_FV_BRISK_00015', 
           'dat_artigo_FV_FREAK_00001', 'dat_artigo_FV_FREAK_00002', 
           'dat_artigo_FV_FREAK_00003', 'dat_artigo_FV_FREAK_00004', 
           'dat_artigo_FV_FREAK_00005', 'dat_artigo_FV_FREAK_00010', 
           'dat_artigo_FV_FREAK_00015', 'dat_artigo_FV__KAZE_00001', 
           'dat_artigo_FV__KAZE_00002', 'dat_artigo_FV__KAZE_00003', 
           'dat_artigo_FV__KAZE_00004', 'dat_artigo_FV__KAZE_00005', 
           'dat_artigo_FV__KAZE_00010', 'dat_artigo_FV__KAZE_00015', 
           'dat_artigo_FV_ORB_00001', 'dat_artigo_FV_ORB_00002', 
           'dat_artigo_FV_ORB_00003', 'dat_artigo_FV_ORB_00004', 
           'dat_artigo_FV_ORB_00005', 'dat_artigo_FV_ORB_00010', 
           'dat_artigo_FV_ORB_00015', 'dat_artigo_FV_SIFT_00001', 
           'dat_artigo_FV_SIFT_00002', 'dat_artigo_FV_SIFT_00003', 
           'dat_artigo_FV_SIFT_00004', 'dat_artigo_FV_SIFT_00005', 
           'dat_artigo_FV_SIFT_00010', 'dat_artigo_FV_SIFT_00015', 
           'dat_artigo_FV_SURF_00001', 'dat_artigo_FV_SURF_00002', 
           'dat_artigo_FV_SURF_00003', 'dat_artigo_FV_SURF_00004', 
           'dat_artigo_FV_SURF_00005', 'dat_artigo_FV_SURF_00010', 
           'dat_artigo_FV_SURF_00015', 'dat_artigo__KAZE', 'dat_artigo_ORB', 
           'dat_artigo_SIFT', 'dat_artigo_SURF' ]
#metodos = ['FV__KAZE_2', 'FV__KAZE_3', 'FV__KAZE_4', 'FV__KAZE_5', 'FV__KAZE_10', 'FV__KAZE_15']
#metodos = ['dat_artigo_BOV__KAZE_00100']
for metodo in metodos:
    
    #metodo = metod[10:]
    path =path_start+metodo+'/'

    config_file = metodo[11:].lower()
    if 'bov' in config_file:
        config_file = 'BOV' + config_file[3:]
    elif 'fv' in config_file:
        config_file = 'FV' + config_file[2:]
        
    if'_50' in metodo[-3:] or metodo[-1:] in 'EFKBT':
        config_file = config_file + '_00000'
    config_file = config_file + '.cfg'

    #print path+config_file
    print path + "cmc1_"+metodo[11:]+".png", os.path.exists(path + "cmc1_"+metodo[11:]+".png")
    
    #if not os.path.exists(path + "cmc1_"+metodo[11:]+".png"):
    if 1==1:    
        
        folds, imagens, gt_filename, sift_folder, sift_type, folds_folder, images_folder, subsets, bag_type, bag_size = ls.le_config(path+config_file)
        
        if "/media" in folds_folder[:6]:
            folds_folder = '/' + folds_folder[10:]
            
        ls.folds_construct(subsets, folds_folder)
        gt_imagens, class_probe, class_gallery = ls.ground_truth(folds_folder, gt_filename)
        cmc = ls.compute_cmc(gt_imagens, path, metodo[11:])
        
        ls.plot_cmc(cmc, metodo[11:], path + "cmc1_"+metodo[11:]+".png", nf=100, dx=10 )
        