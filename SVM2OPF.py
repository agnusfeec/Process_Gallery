# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:56:15 2016

@author: agnus
"""

def le_svm(arq):

    workfile = arq
    rotulos = []
    descritores = []
    
    with open(workfile, 'r') as f:
        for line in f:
             aux = [ x for x in line[:-1].split(' ') ]
             rotulos.append(int(aux[0]))
             aux1 = [ float(x.split(':')[1]) for x in aux[1:] ]
             descritores.append(aux1)
    f.close()

    return rotulos, descritores    

def mapa(rotulos):
    
    aux = sorted(rotulos)
    r = 1
    m = {}
    ant = aux[0]
    m[r] = ant
    for l in aux:
        if l!=ant:
            r = r + 1
            ant = l
            m[l] = r
    
    return m
    
def gravaOPF(arq, rotulos, descritores, mapa):
    
    #
    # <# of samples> <# of labels> <# of features> 
    # <0> <label> <feature 1 from element 0> <feature 2 from element 0> ...
    # <1> <label> <feature 1 from element 1> <feature 2 from element 1> ...
    #

    workfile = arq

    with open(workfile, 'w') as f:   
        f.write(str(len(descritores)) + ' ' + str(len(mapa)) + ' ' + str(len(descritores[0])) + '\n')
        for i in range(len(descritores)):
            f.write(str(i) + ' ' + str(mapa[rotulos[i]]) + ' ')
            for d in descritores[i][:-1]:
                f.write(str(float(d)) + ' ')
            f.write(str(float(descritores[i][-1])) + '\n')
            
    f.close()
    
pasta = '/Projeto/Projetos/dat_artigo_'  #'/home/agnus/Dropbox/LCA/Projeto/tattCBIR/'
arquivo = 'svm_0_training_201608051056.txt'

# ls -R dat_artigo_BOV*/svm_0_test_*

metodos = ['BOV__KAZE_00010', 'BOV__KAZE_00025', 'BOV__KAZE_00050', 'BOV__KAZE_00100', 'BOV__KAZE_00200', 'BOV__KAZE_00500', 'BOV__KAZE_01000', 'BOV_A_KAZE_50_00010', 'BOV_A_KAZE_50_00025', 'BOV_A_KAZE_50_00050', 'BOV_A_KAZE_50_00100', 'BOV_A_KAZE_50_00200', 'BOV_A_KAZE_50_00500', 'BOV_A_KAZE_50_01000', 'BOV_AAKAZE_50_00010', 'BOV_AAKAZE_50_00025', 'BOV_AAKAZE_50_00050', 'BOV_AAKAZE_50_00100', 'BOV_AAKAZE_50_00200', 'BOV_AAKAZE_50_00500', 'BOV_AAKAZE_50_01000', 'BOV_ABRIEF_50_00010', 'BOV_ABRIEF_50_00025', 'BOV_ABRIEF_50_00050', 'BOV_ABRIEF_50_00100', 'BOV_ABRIEF_50_00200', 'BOV_ABRIEF_50_00500', 'BOV_ABRIEF_50_01000', 'BOV_ABRISK_50_00010', 'BOV_ABRISK_50_00025', 'BOV_ABRISK_50_00050', 'BOV_ABRISK_50_00100', 'BOV_ABRISK_50_00200', 'BOV_ABRISK_50_00500', 'BOV_ABRISK_50_01000', 'BOV_AFREAK_50_00010', 'BOV_AFREAK_50_00025', 'BOV_AFREAK_50_00050', 'BOV_AFREAK_50_00100', 'BOV_AFREAK_50_00200', 'BOV_AFREAK_50_00500', 'BOV_AFREAK_50_01000', 'BOV_AKAZE_00010', 'BOV_AKAZE_00025', 'BOV_AKAZE_00050', 'BOV_AKAZE_00100', 'BOV_AKAZE_00200', 'BOV_AKAZE_00500', 'BOV_AKAZE_01000', 'BOV_AORB_50_00010', 'BOV_AORB_50_00025', 'BOV_AORB_50_00050', 'BOV_AORB_50_00100', 'BOV_AORB_50_00200', 'BOV_AORB_50_00500', 'BOV_AORB_50_01000', 'BOV_ASIFT_50_00010', 'BOV_ASIFT_50_00025', 'BOV_ASIFT_50_00050', 'BOV_ASIFT_50_00100', 'BOV_ASIFT_50_00200', 'BOV_ASIFT_50_00500', 'BOV_ASIFT_50_01000', 'BOV_ASURF_50_00010', 'BOV_ASURF_50_00025', 'BOV_ASURF_50_00050', 'BOV_ASURF_50_00100', 'BOV_ASURF_50_00200', 'BOV_ASURF_50_00500', 'BOV_ASURF_50_01000', 'BOV_BRIEF_00010', 'BOV_BRIEF_00025', 'BOV_BRIEF_00050', 'BOV_BRIEF_00100', 'BOV_BRIEF_00200', 'BOV_BRIEF_00500', 'BOV_BRIEF_01000', 'BOV_BRISK_00010', 'BOV_BRISK_00025', 'BOV_BRISK_00050', 'BOV_BRISK_00100', 'BOV_BRISK_00200', 'BOV_BRISK_00500', 'BOV_BRISK_01000', 'BOV_FREAK_00010', 'BOV_FREAK_00025', 'BOV_FREAK_00050', 'BOV_FREAK_00100', 'BOV_FREAK_00200', 'BOV_FREAK_00500', 'BOV_FREAK_01000', 'BOV_ORB_00010', 'BOV_ORB_00025', 'BOV_ORB_00050', 'BOV_ORB_00100', 'BOV_ORB_00200', 'BOV_ORB_00500', 'BOV_ORB_01000', 'BOV_SIFT_00010', 'BOV_SIFT_00025', 'BOV_SIFT_00050', 'BOV_SIFT_00100', 'BOV_SIFT_00200', 'BOV_SIFT_00500', 'BOV_SIFT_01000', 'BOV_SURF_00010', 'BOV_SURF_00025', 'BOV_SURF_00050', 'BOV_SURF_00100', 'BOV_SURF_00200', 'BOV_SURF_00500', 'BOV_SURF_01000', 'FV__KAZE_00001', 'FV__KAZE_00002', 'FV__KAZE_00003', 'FV__KAZE_00004', 'FV__KAZE_00005', 'FV__KAZE_00010', 'FV__KAZE_00015', 'FV_A_KAZE_50_00001', 'FV_A_KAZE_50_00002', 'FV_A_KAZE_50_00003', 'FV_A_KAZE_50_00004', 'FV_A_KAZE_50_00005', 'FV_A_KAZE_50_00010', 'FV_A_KAZE_50_00015', 'FV_AAKAZE_50_00001', 'FV_AAKAZE_50_00002', 'FV_AAKAZE_50_00003', 'FV_AAKAZE_50_00004', 'FV_AAKAZE_50_00005', 'FV_AAKAZE_50_00010', 'FV_AAKAZE_50_00015', 'FV_ABRIEF_50_00001', 'FV_ABRIEF_50_00002', 'FV_ABRIEF_50_00003', 'FV_ABRIEF_50_00004', 'FV_ABRIEF_50_00005', 'FV_ABRIEF_50_00010', 'FV_ABRIEF_50_00015', 'FV_ABRISK_50_00001', 'FV_ABRISK_50_00002', 'FV_ABRISK_50_00003', 'FV_ABRISK_50_00004', 'FV_ABRISK_50_00005', 'FV_ABRISK_50_00010', 'FV_ABRISK_50_00015', 'FV_AFREAK_50_00001', 'FV_AFREAK_50_00002', 'FV_AFREAK_50_00003', 'FV_AFREAK_50_00004', 'FV_AFREAK_50_00005', 'FV_AFREAK_50_00010', 'FV_AFREAK_50_00015', 'FV_AKAZE_00001', 'FV_AKAZE_00002', 'FV_AKAZE_00003', 'FV_AKAZE_00004', 'FV_AKAZE_00005', 'FV_AKAZE_00010', 'FV_AKAZE_00015', 'FV_AORB_50_00001', 'FV_AORB_50_00002', 'FV_AORB_50_00003', 'FV_AORB_50_00004', 'FV_AORB_50_00005', 'FV_AORB_50_00010', 'FV_AORB_50_00015', 'FV_ASIFT_50_00001', 'FV_ASIFT_50_00002', 'FV_ASIFT_50_00003', 'FV_ASIFT_50_00004', 'FV_ASIFT_50_00005', 'FV_ASIFT_50_00010', 'FV_ASIFT_50_00015', 'FV_ASURF_50_00001', 'FV_ASURF_50_00002', 'FV_ASURF_50_00003', 'FV_ASURF_50_00004', 'FV_ASURF_50_00005', 'FV_ASURF_50_00010', 'FV_ASURF_50_00015', 'FV_BRIEF_00001', 'FV_BRIEF_00002', 'FV_BRIEF_00003', 'FV_BRIEF_00004', 'FV_BRIEF_00005', 'FV_BRIEF_00010', 'FV_BRIEF_00015', 'FV_BRISK_00001', 'FV_BRISK_00002', 'FV_BRISK_00003', 'FV_BRISK_00004', 'FV_BRISK_00005', 'FV_BRISK_00010', 'FV_BRISK_00015', 'FV_FREAK_00001', 'FV_FREAK_00002', 'FV_FREAK_00003', 'FV_FREAK_00004', 'FV_FREAK_00005', 'FV_FREAK_00010', 'FV_FREAK_00015', 'FV_ORB_00001', 'FV_ORB_00002', 'FV_ORB_00003', 'FV_ORB_00004', 'FV_ORB_00005', 'FV_ORB_00010', 'FV_ORB_00015', 'FV_SIFT_00001', 'FV_SIFT_00002', 'FV_SIFT_00003', 'FV_SIFT_00004', 'FV_SIFT_00005', 'FV_SIFT_00010', 'FV_SIFT_00015', 'FV_SURF_00001', 'FV_SURF_00002', 'FV_SURF_00003', 'FV_SURF_00004', 'FV_SURF_00005', 'FV_SURF_00010', 'FV_SURF_00015']
seriais = ['201904111037', '201904111039', '201904111040', '201904111041', '201904111043', '201904111044', '201904111047', '201904111525', '201904111535', '201904111545', '201904111555', '201904111606', '201904111618', '201904111633', '201904111942', '201904111948', '201904111955', '201904112001', '201904112009', '201904112018', '201904112028', '201904112359', '201904120002', '201904120006', '201904120009', '201904120013', '201904120018', '201904120024', '201904120115', '201904120133', '201904120150', '201904120206', '201904120225', '201904120246', '201904120311', '201904112211', '201904112216', '201904112221', '201904112226', '201904112233', '201904112239', '201904112247', '201904111124', '201904111125', '201904111126', '201904111127', '201904111128', '201904111130', '201904111131', '201904121515', '201904121519', '201904121523', '201904121527', '201904121532', '201904121537', '201904121544', '201904122153', '201904122229', '201904122306', '201904122341', '201904130020', '201904130100', '201904130145', '201904121642', '201904121653', '201904121703', '201904121714', '201904121727', '201904121740', '201904121756', '201904111154', '201904111154', '201904111155', '201904111155', '201904111156', '201904111157', '201904111158', '201904111203', '201904111207', '201904111210', '201904111213', '201904111217', '201904111221', '201904111226', '201904111141', '201904111142', '201904111142', '201904111143', '201904111144', '201904111145', '201904111146', '201904111258', '201904111259', '201904111259', '201904111259', '201904111300', '201904111300', '201904111301', '201904111328', '201904111333', '201904111339', '201904111344', '201904111351', '201904111357', '201904111405', '201904111304', '201904111306', '201904111307', '201904111309', '201904111310', '201904111312', '201904111315', '201904111050', '201904111051', '201904111051', '201904111052', '201904111053', '201904111053', '201904111054', '201904111651', '201904111656', '201904111700', '201904111705', '201904111709', '201904111713', '201904111718', '201904112041', '201904112044', '201904112046', '201904112049', '201904112051', '201904112054', '201904112056', '201904120033', '201904120034', '201904120035', '201904120037', '201904120038', '201904120039', '201904120040', '201904120341', '201904120347', '201904120354', '201904120400', '201904120406', '201904120413', '201904120908', '201904112258', '201904112300', '201904112302', '201904112304', '201904112306', '201904112308', '201904112310', '201904111134', '201904111135', '201904111135', '201904111135', '201904111136', '201904111136', '201904111137', '201904121554', '201904121555', '201904121556', '201904121558', '201904121559', '201904121600', '201904121602', '201904130238', '201904130255', '201904130312', '201904130329', '201904130346', '201904130402', '201904131019', '201904121816', '201904121821', '201904121826', '201904121831', '201904121836', '201904121841', '201904121846', '201904111200', '201904111200', '201904111200', '201904111200', '201904111201', '201904111201', '201904111201', '201904111233', '201904111234', '201904111236', '201904111237', '201904111238', '201904111239', '201904111240', '201904111149', '201904111149', '201904111149', '201904111150', '201904111150', '201904111150', '201904111151', '201904111302', '201904111302', '201904111303', '201904111303', '201904111303', '201904111303', '201904111303', '201904111414', '201904111417', '201904111419', '201904111421', '201904111424', '201904111426', '201904111429', '201904111318', '201904111318', '201904111319', '201904111320', '201904111320', '201904111321', '201904111322'
           ]
metodos = ['BOV__KAZE_00100_BG000', 'BOV__KAZE_00100_BG025', 'BOV__KAZE_00100_BG050', 'BOV__KAZE_00100_BG075', 'BOV__KAZE_00100_BG100', 'BOV__KAZE_00200_BG000', 'BOV__KAZE_00200_BG025', 'BOV__KAZE_00200_BG050', 'BOV__KAZE_00200_BG075', 'BOV__KAZE_00200_BG100', 'FV_AKAZE_00001_BG000', 'FV_AKAZE_00001_BG025', 'FV_AKAZE_00001_BG050', 'FV_AKAZE_00001_BG075', 'FV_AKAZE_00001_BG100', 'FV__KAZE_00005_BG000', 'FV__KAZE_00005_BG025', 'FV__KAZE_00005_BG050', 'FV__KAZE_00005_BG075', 'FV__KAZE_00005_BG100']
seriais = ['201911062206', '201911062036', '201911062101', '201911062154', '201911062212', '201911070255', '201911070218', '201911070221', '201911070227', '201911070236', '201911062340', '201911062343', '201911062352', '201911070009', '201911070032', '201911070255', '201911070217', '201911070220', '201911070226', '201911070235']

for (metodo,serial) in zip(metodos, seriais):

    print (metodo, serial)
    
    for i in range(5):

        arquivo = 'svm_' + str(i) + '_training_' + serial + '.txt'
        
        rotulos, descritores = le_svm(pasta + metodo + '/' + arquivo)
        m = mapa(rotulos)
        saida = 'opf'+arquivo[3:]
        gravaOPF(pasta + metodo + '/' + saida, rotulos, descritores, m)
        
        arquivo = 'svm_' + str(i) + '_test_' + serial + '.txt'
        
        rotulos, descritores = le_svm(pasta + metodo + '/' + arquivo)
        saida = 'opf'+arquivo[3:]
        gravaOPF(pasta + metodo + '/' + saida, rotulos, descritores, m)
        
