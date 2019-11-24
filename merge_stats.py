#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:39:56 2019

@author: agnus
"""

conj_metodos = [
    'dat_artigo_BOV__KAZE_00100_BG000','dat_artigo_BOV__KAZE_00100_BG025','dat_artigo_BOV__KAZE_00100_BG050','dat_artigo_BOV__KAZE_00100_BG075','dat_artigo_BOV__KAZE_00100_BG100',
    'dat_artigo_BOV__KAZE_00200_BG000','dat_artigo_BOV__KAZE_00200_BG025','dat_artigo_BOV__KAZE_00200_BG050','dat_artigo_BOV__KAZE_00200_BG075','dat_artigo_BOV__KAZE_00200_BG100',
    'dat_artigo_FV_AKAZE_00001_BG000','dat_artigo_FV_AKAZE_00001_BG025','dat_artigo_FV_AKAZE_00001_BG050','dat_artigo_FV_AKAZE_00001_BG075','dat_artigo_FV_AKAZE_00001_BG100',
    'dat_artigo_FV__KAZE_00005_BG000','dat_artigo_FV__KAZE_00005_BG025','dat_artigo_FV__KAZE_00005_BG050','dat_artigo_FV__KAZE_00005_BG075','dat_artigo_FV__KAZE_00005_BG100'
]
    
classifiers = ['opf', 'svm']

file = open('classifiers_stats.txt', 'w')

for classifier in classifiers:

    line = 'Method\tTraing time Value\tTraing time Std\tTesting time Value\t Testing time Std\tAccuracy Value\tAccuracy Std'
    
    for metodo in conj_metodos:    
            
        path = '/Projeto/Projetos/' + metodo + '/'
        arquivo = 'class_' + classifier + '_estats.txt'
        
        line = line + '\n' + metodo[11:] + '\t'
        with open(path + arquivo, 'r') as f:
            for l in f:
                #vl = l[:-1].split('\t')
                line  = line + l[:-1] + '\t'
        f.close()
        
    #print line
    file.write(line)
    
file.close() 