#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:22:46 2019

@author: agnus
"""

import numpy as np
import pandas as pd
from datetime import datetime

pasta = '/Projeto/Projetos/'  #'/home/agnus/Dropbox/LCA/Projeto/tattCBIR/'

# ls -R dat_artigo_BOV*/svm_0_test_*

def compara_metodos(metodos):

    rotulos = [ x[11:] for x in metodos]
    cols=['Mean Train. Time', 'Std. Dev.', 'Mean Test Time', 'Std. Dev.', 'Accuracy','Std. Dev.']
    valores = []

    for metodo in metodos:
        
        #print metodo
        
        workfile = pasta + metodo + '/' + 'class_opf_estats.txt';
        
        with open(workfile, 'r') as f:
    
            valor = []
            for line in f:
                 aux = [ float(x) for x in line[:-1].split('\t') ]
                 valor.append(aux[0])
                 valor.append(aux[1])
                 
            valores.append(valor)
                 
        f.close()
        
    #print valores
    v_opf =  np.asarray(valores[:])
    s = pd.DataFrame(data=valores, index=rotulos, columns=cols)
    
    valores=[]
    for metodo in metodos:
        
        #print metodo
        
        workfile = pasta + metodo + '/' + 'class_svm_estats.txt';
        
        with open(workfile, 'r') as f:
    
            valor = []
            for line in f:
                 aux = [ float(x) for x in line[:-1].split('\t') ]
                 valor.append(aux[0])
                 valor.append(aux[1])
                 
            valores.append(valor)
                 
        f.close()
        
    #print valores
    v_svm =  np.asarray(valores[:])
    s1 = pd.DataFrame(data=valores, index=rotulos, columns=cols)
    #s.plot.bar(figsize=(12,7))
    #s1.plot.bar(figsize=(12,7))
    
    #print v_opf
    ac_opf = v_opf[:,4]
    er_opf = v_opf[:,5]
    #dados1 = pd.DataFrame(data=accur, index=rotulos, columns=['OPF'])
    #print dados1
    #dados1.plot.bar(yerr=erro,figsize=(12,7))
    #print "acuracia"
    #print accur
    #print "std err"
    #print erro
    
    ac_svm = v_svm[:,4]
    er_svm = v_svm[:,5]
    #dados2 = pd.DataFrame(data=accur, index=rotulos, columns=['SVM'])
    #print dados2
    #dados2.plot.bar(yerr=erro,figsize=(12,7))
    #%%
    accur = np.stack((ac_opf, ac_svm), axis=-1)
    erro = np.stack((er_opf, er_svm), axis=-1)
    #print accur
    #print erro
    dados3 = pd.DataFrame(data=accur, index=rotulos, columns=['OPF', 'SVM'])
    dados4 = pd.DataFrame(data=erro, index=rotulos, columns=['OPF', 'SVM'])
    
    #dados3.plot.bar(figsize=(18,9),capsize=4)
    bar = dados3.plot.bar(yerr=dados4,figsize=(12,7),capsize=4, rot=45)
    #dados3.plot.bar(figsize=(12,7))
    fig = bar.get_figure()
    DT = (datetime.now()).strftime("%Y%m%d%H%M%S")
    p = metodos[0].find('_0')
    fig.savefig('clas_OPF_SVM_'+metodos[0][11:p]+'.png', dpi=300, bbox_inches='tight')

conj_metodos = [
['dat_artigo_BOV__KAZE_00010', 'dat_artigo_BOV__KAZE_00025', 'dat_artigo_BOV__KAZE_00050', 'dat_artigo_BOV__KAZE_00100', 'dat_artigo_BOV__KAZE_00200', 'dat_artigo_BOV__KAZE_00500', 'dat_artigo_BOV__KAZE_01000'],
['dat_artigo_BOV_AKAZE_00010', 'dat_artigo_BOV_AKAZE_00025', 'dat_artigo_BOV_AKAZE_00050', 'dat_artigo_BOV_AKAZE_00100', 'dat_artigo_BOV_AKAZE_00200', 'dat_artigo_BOV_AKAZE_00500', 'dat_artigo_BOV_AKAZE_01000'],
['dat_artigo_BOV_BRIEF_00010', 'dat_artigo_BOV_BRIEF_00025', 'dat_artigo_BOV_BRIEF_00050', 'dat_artigo_BOV_BRIEF_00100', 'dat_artigo_BOV_BRIEF_00200', 'dat_artigo_BOV_BRIEF_00500', 'dat_artigo_BOV_BRIEF_01000'],
['dat_artigo_BOV_BRISK_00010', 'dat_artigo_BOV_BRISK_00025', 'dat_artigo_BOV_BRISK_00050', 'dat_artigo_BOV_BRISK_00100', 'dat_artigo_BOV_BRISK_00200', 'dat_artigo_BOV_BRISK_00500', 'dat_artigo_BOV_BRISK_01000'],
['dat_artigo_BOV_FREAK_00010', 'dat_artigo_BOV_FREAK_00025', 'dat_artigo_BOV_FREAK_00050', 'dat_artigo_BOV_FREAK_00100', 'dat_artigo_BOV_FREAK_00200', 'dat_artigo_BOV_FREAK_00500', 'dat_artigo_BOV_FREAK_01000'],
['dat_artigo_BOV_ORB_00010', 'dat_artigo_BOV_ORB_00025', 'dat_artigo_BOV_ORB_00050', 'dat_artigo_BOV_ORB_00100', 'dat_artigo_BOV_ORB_00200', 'dat_artigo_BOV_ORB_00500', 'dat_artigo_BOV_ORB_01000'],
['dat_artigo_BOV_SURF_00010', 'dat_artigo_BOV_SURF_00025', 'dat_artigo_BOV_SURF_00050', 'dat_artigo_BOV_SURF_00100', 'dat_artigo_BOV_SURF_00200', 'dat_artigo_BOV_SURF_00500', 'dat_artigo_BOV_SURF_01000'],
['dat_artigo_BOV_SIFT_00010', 'dat_artigo_BOV_SIFT_00025', 'dat_artigo_BOV_SIFT_00050', 'dat_artigo_BOV_SIFT_00100', 'dat_artigo_BOV_SIFT_00200', 'dat_artigo_BOV_SIFT_00500', 'dat_artigo_BOV_SIFT_01000'],
['dat_artigo_BOV_A_KAZE_50_00010', 'dat_artigo_BOV_A_KAZE_50_00025', 'dat_artigo_BOV_A_KAZE_50_00050', 'dat_artigo_BOV_A_KAZE_50_00100', 'dat_artigo_BOV_A_KAZE_50_00200', 'dat_artigo_BOV_A_KAZE_50_00500', 'dat_artigo_BOV_A_KAZE_50_01000'],
['dat_artigo_BOV_AAKAZE_50_00010', 'dat_artigo_BOV_AAKAZE_50_00025', 'dat_artigo_BOV_AAKAZE_50_00050', 'dat_artigo_BOV_AAKAZE_50_00100', 'dat_artigo_BOV_AAKAZE_50_00200', 'dat_artigo_BOV_AAKAZE_50_00500', 'dat_artigo_BOV_AAKAZE_50_01000'],
['dat_artigo_BOV_ABRIEF_50_00010', 'dat_artigo_BOV_ABRIEF_50_00025', 'dat_artigo_BOV_ABRIEF_50_00050', 'dat_artigo_BOV_ABRIEF_50_00100', 'dat_artigo_BOV_ABRIEF_50_00200', 'dat_artigo_BOV_ABRIEF_50_00500', 'dat_artigo_BOV_ABRIEF_50_01000'],
['dat_artigo_BOV_ABRISK_50_00010', 'dat_artigo_BOV_ABRISK_50_00025', 'dat_artigo_BOV_ABRISK_50_00050', 'dat_artigo_BOV_ABRISK_50_00100', 'dat_artigo_BOV_ABRISK_50_00200', 'dat_artigo_BOV_ABRISK_50_00500', 'dat_artigo_BOV_ABRISK_50_01000'],
['dat_artigo_BOV_AFREAK_50_00010', 'dat_artigo_BOV_AFREAK_50_00025', 'dat_artigo_BOV_AFREAK_50_00050', 'dat_artigo_BOV_AFREAK_50_00100', 'dat_artigo_BOV_AFREAK_50_00200', 'dat_artigo_BOV_AFREAK_50_00500', 'dat_artigo_BOV_AFREAK_50_01000'],
['dat_artigo_BOV_AORB_50_00010', 'dat_artigo_BOV_AORB_50_00025', 'dat_artigo_BOV_AORB_50_00050', 'dat_artigo_BOV_AORB_50_00100', 'dat_artigo_BOV_AORB_50_00200', 'dat_artigo_BOV_AORB_50_00500', 'dat_artigo_BOV_AORB_50_01000'],
['dat_artigo_BOV_ASURF_50_00010', 'dat_artigo_BOV_ASURF_50_00025', 'dat_artigo_BOV_ASURF_50_00050', 'dat_artigo_BOV_ASURF_50_00100', 'dat_artigo_BOV_ASURF_50_00200', 'dat_artigo_BOV_ASURF_50_00500', 'dat_artigo_BOV_ASURF_50_01000'],
['dat_artigo_BOV_ASIFT_50_00010', 'dat_artigo_BOV_ASIFT_50_00025', 'dat_artigo_BOV_ASIFT_50_00050', 'dat_artigo_BOV_ASIFT_50_00100', 'dat_artigo_BOV_ASIFT_50_00200', 'dat_artigo_BOV_ASIFT_50_00500', 'dat_artigo_BOV_ASIFT_50_01000'],
['dat_artigo_FV__KAZE_00001', 'dat_artigo_FV__KAZE_00002', 'dat_artigo_FV__KAZE_00003', 'dat_artigo_FV__KAZE_00004', 'dat_artigo_FV__KAZE_00005', 'dat_artigo_FV__KAZE_00010', 'dat_artigo_FV__KAZE_00015'], 
['dat_artigo_FV_AKAZE_00001', 'dat_artigo_FV_AKAZE_00002', 'dat_artigo_FV_AKAZE_00003', 'dat_artigo_FV_AKAZE_00004', 'dat_artigo_FV_AKAZE_00005', 'dat_artigo_FV_AKAZE_00010', 'dat_artigo_FV_AKAZE_00015'], 
['dat_artigo_FV_BRIEF_00001', 'dat_artigo_FV_BRIEF_00002', 'dat_artigo_FV_BRIEF_00003', 'dat_artigo_FV_BRIEF_00004', 'dat_artigo_FV_BRIEF_00005', 'dat_artigo_FV_BRIEF_00010', 'dat_artigo_FV_BRIEF_00015'], 
['dat_artigo_FV_BRISK_00001', 'dat_artigo_FV_BRISK_00002', 'dat_artigo_FV_BRISK_00003', 'dat_artigo_FV_BRISK_00004', 'dat_artigo_FV_BRISK_00005', 'dat_artigo_FV_BRISK_00010', 'dat_artigo_FV_BRISK_00015'], 
['dat_artigo_FV_FREAK_00001', 'dat_artigo_FV_FREAK_00002', 'dat_artigo_FV_FREAK_00003', 'dat_artigo_FV_FREAK_00004', 'dat_artigo_FV_FREAK_00005', 'dat_artigo_FV_FREAK_00010', 'dat_artigo_FV_FREAK_00015'], 
['dat_artigo_FV_ORB_00001', 'dat_artigo_FV_ORB_00002', 'dat_artigo_FV_ORB_00003', 'dat_artigo_FV_ORB_00004', 'dat_artigo_FV_ORB_00005', 'dat_artigo_FV_ORB_00010', 'dat_artigo_FV_ORB_00015'], 
['dat_artigo_FV_SIFT_00001', 'dat_artigo_FV_SIFT_00002', 'dat_artigo_FV_SIFT_00003', 'dat_artigo_FV_SIFT_00004', 'dat_artigo_FV_SIFT_00005', 'dat_artigo_FV_SIFT_00010', 'dat_artigo_FV_SIFT_00015'], 
['dat_artigo_FV_SURF_00001', 'dat_artigo_FV_SURF_00002', 'dat_artigo_FV_SURF_00003', 'dat_artigo_FV_SURF_00004', 'dat_artigo_FV_SURF_00005', 'dat_artigo_FV_SURF_00010', 'dat_artigo_FV_SURF_00015'],
['dat_artigo_FV_A_KAZE_50_00001', 'dat_artigo_FV_A_KAZE_50_00002', 'dat_artigo_FV_A_KAZE_50_00003', 'dat_artigo_FV_A_KAZE_50_00004', 'dat_artigo_FV_A_KAZE_50_00005', 'dat_artigo_FV_A_KAZE_50_00010', 'dat_artigo_FV_A_KAZE_50_00015'], 
['dat_artigo_FV_AAKAZE_50_00001', 'dat_artigo_FV_AAKAZE_50_00002', 'dat_artigo_FV_AAKAZE_50_00003', 'dat_artigo_FV_AAKAZE_50_00004', 'dat_artigo_FV_AAKAZE_50_00005', 'dat_artigo_FV_AAKAZE_50_00010', 'dat_artigo_FV_AAKAZE_50_00015'], 
['dat_artigo_FV_ABRIEF_50_00001', 'dat_artigo_FV_ABRIEF_50_00002', 'dat_artigo_FV_ABRIEF_50_00003', 'dat_artigo_FV_ABRIEF_50_00004', 'dat_artigo_FV_ABRIEF_50_00005', 'dat_artigo_FV_ABRIEF_50_00010', 'dat_artigo_FV_ABRIEF_50_00015'], 
['dat_artigo_FV_ABRISK_50_00001', 'dat_artigo_FV_ABRISK_50_00002', 'dat_artigo_FV_ABRISK_50_00003', 'dat_artigo_FV_ABRISK_50_00004', 'dat_artigo_FV_ABRISK_50_00005', 'dat_artigo_FV_ABRISK_50_00010', 'dat_artigo_FV_ABRISK_50_00015'], 
['dat_artigo_FV_AFREAK_50_00001', 'dat_artigo_FV_AFREAK_50_00002', 'dat_artigo_FV_AFREAK_50_00003', 'dat_artigo_FV_AFREAK_50_00004', 'dat_artigo_FV_AFREAK_50_00005', 'dat_artigo_FV_AFREAK_50_00010', 'dat_artigo_FV_AFREAK_50_00015'], 
['dat_artigo_FV_AORB_50_00001', 'dat_artigo_FV_AORB_50_00002', 'dat_artigo_FV_AORB_50_00003', 'dat_artigo_FV_AORB_50_00004', 'dat_artigo_FV_AORB_50_00005', 'dat_artigo_FV_AORB_50_00010', 'dat_artigo_FV_AORB_50_00015'], 
['dat_artigo_FV_ASIFT_50_00001', 'dat_artigo_FV_ASIFT_50_00002', 'dat_artigo_FV_ASIFT_50_00003', 'dat_artigo_FV_ASIFT_50_00004', 'dat_artigo_FV_ASIFT_50_00005', 'dat_artigo_FV_ASIFT_50_00010', 'dat_artigo_FV_ASIFT_50_00015'], 
['dat_artigo_FV_ASURF_50_00001', 'dat_artigo_FV_ASURF_50_00002', 'dat_artigo_FV_ASURF_50_00003', 'dat_artigo_FV_ASURF_50_00004', 'dat_artigo_FV_ASURF_50_00005', 'dat_artigo_FV_ASURF_50_00010', 'dat_artigo_FV_ASURF_50_00015'],
   ]

conj_metodos = [
['dat_artigo_BOV__KAZE_00200', 'dat_artigo_BOV_AKAZE_00050', 'dat_artigo_BOV_BRIEF_00025', 'dat_artigo_BOV_BRISK_00050', 'dat_artigo_BOV_FREAK_00050', 'dat_artigo_BOV_ORB_00200', 'dat_artigo_BOV_SIFT_00200', 'dat_artigo_BOV_SURF_00100'],
['dat_artigo_BOV_A_KAZE_50_00100', 'dat_artigo_BOV_AAKAZE_50_00050', 'dat_artigo_BOV_ABRIEF_50_00100', 'dat_artigo_BOV_ABRISK_50_00100', 'dat_artigo_BOV_AFREAK_50_00100', 'dat_artigo_BOV_AORB_50_00100', 'dat_artigo_BOV_ASIFT_50_00500', 'dat_artigo_BOV_ASURF_50_00100'],
['dat_artigo_FV__KAZE_00005', 'dat_artigo_FV_AKAZE_00002', 'dat_artigo_FV_BRIEF_00003', 'dat_artigo_FV_BRISK_00001', 'dat_artigo_FV_FREAK_00001', 'dat_artigo_FV_ORB_00010', 'dat_artigo_FV_SIFT_00001', 'dat_artigo_FV_SURF_00005'],
['dat_artigo_FV_A_KAZE_50_00004', 'dat_artigo_FV_AAKAZE_50_00005', 'dat_artigo_FV_ABRIEF_50_00004', 'dat_artigo_FV_ABRISK_50_00004', 'dat_artigo_FV_AFREAK_50_00002', 'dat_artigo_FV_AORB_50_00005', 'dat_artigo_FV_ASIFT_50_00005', 'dat_artigo_FV_ASURF_50_00015']
]

conj_metodos = [
    ['dat_artigo_BOV__KAZE_00100_BG000','dat_artigo_BOV__KAZE_00100_BG025','dat_artigo_BOV__KAZE_00100_BG050','dat_artigo_BOV__KAZE_00100_BG075','dat_artigo_BOV__KAZE_00100_BG100'],
    ['dat_artigo_BOV__KAZE_00200_BG000','dat_artigo_BOV__KAZE_00200_BG025','dat_artigo_BOV__KAZE_00200_BG050','dat_artigo_BOV__KAZE_00200_BG075','dat_artigo_BOV__KAZE_00200_BG100'],
    ['dat_artigo_FV_AKAZE_00001_BG000','dat_artigo_FV_AKAZE_00001_BG025','dat_artigo_FV_AKAZE_00001_BG050','dat_artigo_FV_AKAZE_00001_BG075','dat_artigo_FV_AKAZE_00001_BG100'],
    ['dat_artigo_FV__KAZE_00005_BG000','dat_artigo_FV__KAZE_00005_BG025','dat_artigo_FV__KAZE_00005_BG050','dat_artigo_FV__KAZE_00005_BG075','dat_artigo_FV__KAZE_00005_BG100']
]
    
for metodos in conj_metodos:
    #build_cmc(arquivos, mylabels, mystyle, linestyles)
    print metodos
    compara_metodos(metodos)