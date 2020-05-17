#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:24:05 2019

@author: agnus
"""

def build_cmc(arquivos, mylabels, mystyle, linestyles, metodos):

    #import matplotlib.pyplot as plt
    import pylab as P
    import numpy as np
    import matplotlib
    from datetime import datetime

    matplotlib.rcParams.update({'font.size': 17})

    fig = P.figure()
    fig.suptitle('Cumulative Matching Characteristic', fontsize=18, fontweight='bold')
    #pos = metodos[0].find('0')
    #P.title(metodos[0][11:(pos-1)])

    
    P.ylabel('%', fontsize=16)
    P.xlabel('Rank', fontsize=16)
    ni = 50 #300
    P.xlim(0, ni)
    P.ylim(0,101)
    P.xticks(np.arange(0, ni, 10.0))
    P.yticks(np.arange(0, 101, 5.0))

    xticklabels = P.getp(P.gca(), 'xticklabels')
    yticklabels = P.getp(P.gca(), 'yticklabels')

    P.setp(yticklabels, 'color', 'k', fontsize='x-large')
    P.setp(xticklabels, 'color', 'k', fontsize='x-large')

    P.grid(True)
    fig.set_size_inches(10,7)

    for (aq,pt,ls) in zip(arquivos,mystyle,linestyles):
        #print aq
        #cmc = np.loadtxt('./BOV_' + aq + '_ASIFT/cmc_bov_' + aq + '_asift.out',delimiter=',')
        #cmc = np.loadtxt('./FV_' + aq + 'GMM_SIFT/cmc_fv_' + aq + '_sift.out',delimiter=',')
        cmc = np.loadtxt(aq[0]+aq[1], delimiter=',')
        P.plot(cmc*100, pt, label= aq, linewidth=1, ms=10, markevery=5) #, linestyle=ls) #[:-6])
    P.legend(bbox_to_anchor=(0.95,0.5), borderaxespad=0.0, labels=mylabels )

    #fig.savefig('cmc_SIFT_ASIFT_MHSLCD.png')

    DT = (datetime.now()).strftime("%Y%m%d%H%M%S")

    fig.set_size_inches(12, 8)
    pos = mylabels[0].find('0')
    if pos<=0:
        pos =len(metodos[0])+1
    fig.savefig('cmc_Mescla_'+DT+'.png', dpi=300, bbox_inches='tight')
    #fig.savefig('cmc_SIFT_ASIFT_MHSLCD.png', dpi=300, bbox_inches='tight')
    #fig.savefig('cmc_FV_SIFT.png', dpi=300, bbox_inches='tight')
    #fig.savefig('cmc_BOV_ASIFT.png', dpi=300, bbox_inches='tight')

    P.show()

def construct_data(metodos):
    arquivos = []
    mylabels = []
    for metodo in metodos:

        cmc_dat = 'cmc_'+metodo[11:]+'.dat'
        path =path_start+'dat_artigo_'+metodo[11:]+'/'

        #print cmc_dat
        #print path

        arquivos.append((path,cmc_dat))
        mylabels.append(metodo[11:])

    #myPath = '/media/sf_Projeto/dados_artigo_atualizado/graficos'
    mystyle = ['r-s','g->','b-v','y-^','m-o', 'c-d','b-s','r->','b-v','y-^']
    linestyles = ['-', '--', '-.', ':', '-', '--',':','-', '--', '-.', ':']

    return arquivos, mylabels, mystyle,linestyles


#%%
path_start = "/Projeto/Projetos/"


conj_metodos = [
['dat_artigo__KAZE', 'dat_artigo_AKAZE', 'dat_artigo_BRIEF', 'dat_artigo_BRISK', 'dat_artigo_FREAK', 'dat_artigo_ORB', 'dat_artigo_SIFT', 'dat_artigo_SURF'],
['dat_artigo_A_KAZE_50', 'dat_artigo_AAKAZE_50', 'dat_artigo_ABRIEF_50', 'dat_artigo_ABRISK_50', 'dat_artigo_AFREAK_50', 'dat_artigo_AORB_50', 'dat_artigo_ASIFT_50', 'dat_artigo_ASURF_50'],
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

# MESCLA OS MELHORES RESULTADOS
conj_metodos =[
['dat_artigo_SIFT', 'dat_artigo_ASIFT_50', 'dat_artigo_BOV__KAZE_00100', 'dat_artigo_BOV_A_KAZE_50_00200', 'dat_artigo_FV__KAZE_00003', 'dat_artigo_FV_A_KAZE_50_00005'], 
['dat_artigo_SIFT', 'dat_artigo_ASIFT_50', 'dat_artigo_BOV_AKAZE_00025', 'dat_artigo_BOV_AAKAZE_50_00050', 'dat_artigo_FV_AKAZE_00001', 'dat_artigo_FV_AAKAZE_50_00003'], 
['dat_artigo_SIFT', 'dat_artigo_ASIFT_50', 'dat_artigo_BOV_BRIEF_00025', 'dat_artigo_BOV_ABRIEF_50_00050', 'dat_artigo_FV_BRIEF_00002', 'dat_artigo_FV_ABRIEF_50_00003'], 
['dat_artigo_SIFT', 'dat_artigo_ASIFT_50', 'dat_artigo_BOV_BRISK_00050', 'dat_artigo_BOV_ABRISK_50_00100', 'dat_artigo_FV_BRISK_00001', 'dat_artigo_FV_ABRISK_50_00003'], 
['dat_artigo_SIFT', 'dat_artigo_ASIFT_50', 'dat_artigo_BOV_FREAK_00010', 'dat_artigo_BOV_AFREAK_50_00050', 'dat_artigo_FV_FREAK_00001', 'dat_artigo_FV_AFREAK_50_00003'], 
['dat_artigo_SIFT', 'dat_artigo_ASIFT_50', 'dat_artigo_BOV_ORB_00200', 'dat_artigo_BOV_AORB_50_00200', 'dat_artigo_FV_ORB_00005', 'dat_artigo_FV_AORB_50_00005'], 
['dat_artigo_SIFT', 'dat_artigo_ASIFT_50', 'dat_artigo_BOV_SURF_00050', 'dat_artigo_BOV_ASURF_50_00050', 'dat_artigo_FV_SURF_00003', 'dat_artigo_FV_ASURF_50_00010'], 
['dat_artigo_SIFT', 'dat_artigo_ASIFT_50', 'dat_artigo_BOV_SIFT_00050', 'dat_artigo_BOV_ASIFT_50_00200', 'dat_artigo_FV_SIFT_00001', 'dat_artigo_FV_ASIFT_50_00005']
]
conj_metodos = [['dat_artigo_SIFT_00000_BG000','dat_artigo_SIFT_00000_BG025','dat_artigo_SIFT_00000_BG050','dat_artigo_SIFT_00000_BG075','dat_artigo_SIFT_00000_BG100']]

conj_metodos = [['dat_artigo_AKAZE_00000_BG000','dat_artigo_AKAZE_00000_BG025','dat_artigo_AKAZE_00000_BG050','dat_artigo_AKAZE_00000_BG075','dat_artigo_AKAZE_00000_BG100']]

conj_metodos =[['dat_artigo_BOV__KAZE_00100_BG000','dat_artigo_BOV__KAZE_00100_BG025','dat_artigo_BOV__KAZE_00100_BG050','dat_artigo_BOV__KAZE_00100_BG075','dat_artigo_BOV__KAZE_00100_BG100']]

conj_metodos =[['dat_artigo_BOV__KAZE_00200_BG000','dat_artigo_BOV__KAZE_00200_BG025','dat_artigo_BOV__KAZE_00200_BG050','dat_artigo_BOV__KAZE_00200_BG075','dat_artigo_BOV__KAZE_00200_BG100']]

conj_metodos = [['dat_artigo_FV_AKAZE_00001_BG000','dat_artigo_FV_AKAZE_00001_BG025','dat_artigo_FV_AKAZE_00001_BG050','dat_artigo_FV_AKAZE_00001_BG075','dat_artigo_FV_AKAZE_00001_BG100']]

conj_metodos = [['dat_artigo_FV__KAZE_00005_BG000','dat_artigo_FV__KAZE_00005_BG025','dat_artigo_FV__KAZE_00005_BG050','dat_artigo_FV__KAZE_00005_BG075','dat_artigo_FV__KAZE_00005_BG100']]

conj_metodos = [['dat_artigo__KAZE_00000_BG000','dat_artigo__KAZE_00000_BG025','dat_artigo__KAZE_00000_BG050','dat_artigo__KAZE_00000_BG075','dat_artigo__KAZE_00000_BG100']]

conj_metodos = [['dat_artigo_BOV_SIFT_25', 'dat_artigo_BOV_SIFT_50', 'dat_artigo_BOV_SIFT_100', 'dat_artigo_BOV_SIFT_200', 'dat_artigo_BOV_SIFT_250','dat_artigo_BOV_SIFT_500', 'dat_artigo_BOV_SIFT_1000']]
conj_metodos = [['dat_artigo_BOV_SURF_25', 'dat_artigo_BOV_SURF_50', 'dat_artigo_BOV_SURF_100', 'dat_artigo_BOV_SURF_200', 'dat_artigo_BOV_SURF_250','dat_artigo_BOV_SURF_500', 'dat_artigo_BOV_SURF_1000']]
conj_metodos = [['dat_artigo_BOV_ORB_25', 'dat_artigo_BOV_ORB_50', 'dat_artigo_BOV_ORB_100', 'dat_artigo_BOV_ORB_200', 'dat_artigo_BOV_ORB_250','dat_artigo_BOV_ORB_500', 'dat_artigo_BOV_ORB_1000']]
conj_metodos = [['dat_artigo_BOV__KAZE_25', 'dat_artigo_BOV__KAZE_50', 'dat_artigo_BOV__KAZE_100', 'dat_artigo_BOV__KAZE_200', 'dat_artigo_BOV__KAZE_250','dat_artigo_BOV__KAZE_500', 'dat_artigo_BOV__KAZE_1000']]
conj_metodos = [[ 'dat_artigo_BOV_FREAK_50', 'dat_artigo_BOV_FREAK_100', 'dat_artigo_BOV_FREAK_200', 'dat_artigo_BOV_FREAK_250','dat_artigo_BOV_FREAK_500', 'dat_artigo_BOV_FREAK_1000']]

conj_metodos = [['dat_artigo__KAZE_00000_BG000', 'dat_artigo_AKAZE_00000_BG000', 'dat_artigo_BRIEF_00000_BG000', 'dat_artigo_BRISK_00000_BG000', 'dat_artigo_FREAK_00000_BG000', 'dat_artigo_ORB_00000_BG000', 'dat_artigo_SIFT_00000_BG000', 'dat_artigo_SURF_00000_BG000'],
['dat_artigo_A_KAZE_050__00000_BG000', 'dat_artigo_AAKAZE_050__00000_BG000', 'dat_artigo_ABRIEF_050__00000_BG000', 'dat_artigo_ABRISK_050__00000_BG000', 'dat_artigo_AFREAK_050__00000_BG000', 'dat_artigo_AORB_050__00000_BG000', 'dat_artigo_ASIFT_050__00000_BG000', 'dat_artigo_ASURF_050__00000_BG000']]

#  conj_metodos = [['dat_artigo_BOV_SIFT_50','dat_artigo_BOV_SURF_100','dat_artigo_BOV_ORB_250', 'dat_artigo_BOV__KAZE_250', 'dat_artigo_BOV_FREAK_25',
#conj_metodos = [
#['dat_artigo_BOV_A_KAZE_50_00010', 'dat_artigo_BOV_A_KAZE_50_00025', 'dat_artigo_BOV_A_KAZE_50_00050', 'dat_artigo_BOV_A_KAZE_50_00100', 'dat_artigo_BOV_A_KAZE_50_00200', 'dat_artigo_BOV_A_KAZE_50_00500', 'dat_artigo_BOV_A_KAZE_50_01000']
#        ]
#%%
for metodos in conj_metodos:
    #build_cmc(arquivos, mylabels, mystyle, linestyles)
    print metodos
    arquivos, mylabels, mystyle,linestyles = construct_data(metodos)
    build_cmc(arquivos, mylabels, mystyle,linestyles, metodos)

