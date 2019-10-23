# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:20:08 2016

@author: agnus
"""

from datetime import datetime
import time
import logging
import numpy as np
import os

import lib_sistema_02 as ls

#%%
def rodada(filename, metodo, bag, size, n_subsets, bg):

    path = "/Projeto/Projetos/dat_artigo"
    print (path)
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    path = path + "/"
    
    # Pega a hora do sistema para identificar as diferentes execuções
    DT = (datetime.now()).strftime("%Y%m%d%H%M")
    
    # Define o sitema de log para registrar as ocorrências
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)
    
    # create a file HANDLER
    HANDLER = logging.FileHandler(path + filename +'.log')
    HANDLER.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    HANDLER.setFormatter(formatter)
    
    # add the HANDLERs to the LOGGER
    LOGGER.addHandler(HANDLER)
    
    #%%
    arquivo = '/Projeto/Projetos/dat_artigo/' + filename + '.cfg'
    
    ls.grava_config(arquivo, metodo, bag, size, n_subsets, bg)
    folds, imagens, gt_filename, sift_folder, sift_type, folds_folder, images_folder, subsets, bag_type, bag_size, bg_perc = ls.le_config(arquivo)
    
    t_start = time.time()
    LOGGER.info('folds_construct: starting')
    folds = ls.folds_construct(subsets, folds_folder, bg_perc)
    LOGGER.info('folds_construct: ending(' + str(time.time()-t_start)+')')
    
    t_start = time.time()
    LOGGER.info('ground_truth: starting')
    gt_imagens, class_probe, class_gallery = ls.ground_truth(folds_folder, gt_filename)
    LOGGER.info('ground_truth: ending(' + str(time.time()-t_start)+')')
    
    #t_start = time.time()
    #LOGGER.info('gera_sift_base: starting')
    #ls.gera_sift_base(folds, imagens, sift_folder, sift_type)
    #LOGGER.info('gera_sift_base: ending(' + str(time.time()-t_start)+')')
    
    if "NONE" in bag_type.upper() :
        metodo = sift_type.upper() + "_" + str(bag_size).zfill(5) + '_BG' + str(bg).zfill(3)
    else:
        metodo = bag_type + "_" + sift_type.upper() + "_" + str(bag_size).zfill(5)+ '_BG' + str(bg).zfill(3)
    
    #metodo = "ASIFT_Ampliado"
    
    #metodos = ["SIFT", "ASIFT", "BRISK", "SURF", "BRIEF", "ORB", "FREAK", "AKAZE", "_AKAZE"]
    
    if ("SIFT" in metodo or "BRISK" in metodo or "SURF" in metodo or "BRIEF" in metodo or "ORB" in metodo or "FREAK" in metodo or "AKAZE" in metodo or "_AKAZE")  and not ("BOV" in metodo) and not ("FV" in metodo):
    
        t_start = time.time()
        LOGGER.info('processa_' + metodo + ': starting')
        ls.processa_sift(folds, imagens, sift_folder, sift_type, LOGGER, metodo)
        LOGGER.info('processa_' + metodo + ': ending(' + str(time.time()-t_start)+')')
    
    elif "FV" in metodo:
    
        # Inicialmente esta considerando apenas um fold, deve ser verifcado o caso
        # de ter mais de um fold
    
        n_folds = len(folds) # por enquanto n_folds será 1 pois tem apenas um fold
        for i in range(n_folds):
            
            t_start_folder = time.time()
            LOGGER.info('processa_' + metodo + ' Folder ' + str(i+1) + ': starting')
            
            train = folds[i][0]
            for j in range(n_folds):
                if j != i:
                    train = train + folds[j][0]+folds[j][1] #+folds[j][2]
    
            t_start = time.time()
            LOGGER.info('le_descritores_train: starting')
            ds, id_ds = ls.le_descritores(sift_folder, sift_type, train, tipo=1)
            LOGGER.info('le_descritores_train: ending(' + str(time.time()-t_start)+')')
    
    #%%
            N = bag_size # incluir posteriormente no arquivo de configuração
            t_start = time.time()
            LOGGER.info('fv_generate_gmm: starting')
            gmm = ls.fv_generate_gmm(i, ds, N, DT)
            LOGGER.info('fv_generate_gmm: ending(' + str(time.time()-t_start)+')')
    
    #%%
            # # change to allow background images to be included
            if len(subsets)>2 :
                train = train+folds[i][2]
    
                if i==0: # Read data from background images only in the first time
                    t_start = time.time()
                    LOGGER.info('le_descritores_train expandido: starting')
                    ds1, id_ds1 = ls.le_descritores(sift_folder, sift_type, folds[i][2], 1)
                    LOGGER.info('le_descritores_train expandido: ending(' + str(time.time()-t_start)+')')
    
                ds = ds + ds1
                id_ds.extend(id_ds1)

        #%%%
    
            #codifica a base em função das gmm treinadas
            t_start = time.time()
            LOGGER.info('fv_fisher_vector for train: starting [size = ' + str(len(ds)) + ']')
            fv_train = np.float32([ls.fv_fisher_vector(descriptor, *gmm) for descriptor in ds])
            LOGGER.info('fv_fisher_vector for train: ending(' + str(time.time()-t_start)+')')
    
            #X_train = fv_train
    
            #%%
    
            #t_start = time.time()
            #LOGGER.info('FV_vetores_grava: starting')
            #ls.bov_histogramas_grava(fv_train, DT)
            #LOGGER.info('FV_vetores_grava: ending(' + str(time.time()-t_start)+')')
    
    #%%
            classes = []
            for image in train:
                if class_gallery.has_key(image):
                    classes.append(class_gallery[image])
                else:
                    classes.append(class_probe[image])
    
            ls.gera_svm_dat(i, classes, fv_train, DT, False)
    
    #%%
            # Inicialmente esta considerando apenas um fold, deve ser verifcado o caso
            # de ter mais de um fold
    
            #n_folds = len(folds) # por enquanto n_folds será 1 pois tem apenas um fold
            #for i in range(n_folds):
            test = folds[i][1]
    
            #ds = ls.le_descritores(sift_folder, test)
            t_start = time.time()
            LOGGER.info('le_descritores_test: starting')
            ds, id_ds = ls.le_descritores(sift_folder, sift_type, test)
            LOGGER.info('le_descritores_test: ending(' + str(time.time()-t_start)+')')
    
    #%%
            #codifica o conjunto de teste em função das gmm treinadas
            t_start = time.time()
            LOGGER.info('fv_fisher_vector for test: starting')
            fv_test = np.float32([ls.fv_fisher_vector(descriptor, *gmm) for descriptor in ds])
            LOGGER.info('fv_fisher_vector for test: ending(' + str(time.time()-t_start)+')')
    
            #X_test = fv_test
    
    #%%
            classes = []
            for image in test:
                classes.append(class_probe[image])
    
            ls.gera_svm_dat(i, classes, fv_test, DT, True)
    
    #%%
            import scipy.spatial.distance as ssd
    
            # não está considerando a questão de multiplos folds
            # e também o uso de memória auxiliar no disco
    
            ntrain = fv_train.shape[0]
    
            #i = 0
            #path = "/media/agnus/My Passport/Projeto/dat_artigo/"
    
            arquivo = path + 'clist_mem_'+str(i+1)+'.txt'
            with open(arquivo, 'w') as clist_file:
    
                ntest = fv_test.shape[0]
    
                for i_test in range(ntest):
    
                    file_test = test[i_test]
                    u = fv_test[i_test]
                    dist = np.zeros((ntrain))
    
                    for i_train in range(ntrain):
    
                        v = fv_train[i_train]
                        #dist[i_train] = ssd.cityblock(u, v)
                        dist[i_train] = ssd.euclidean(u, v)
    
                    #indice = np.argsort(dist)[::-1]
                    indice = np.argsort(dist)
    
                    k = 1
                    for idx in indice:
                        clist_file.write(file_test+'|'+ str(k) +
                                         '|' + train[idx] + '|' + str(dist[idx]) +'\n')
                        k = k + 1
    
            LOGGER.info('processa_' + metodo + ' Folder : ending(' + str(time.time()-t_start_folder)+')')
            
    elif "BOV" in metodo:
    
    #%%
        # Inicialmente esta considerando apenas um fold, deve ser verifcado o caso
        # de ter mais de um fold
    
        n_folds = len(folds) # por enquanto n_folds será 1 pois tem apenas um fold
        for i in range(n_folds):
            
            t_start_folder = time.time()
            LOGGER.info('processa_' + metodo + ' Folder ' + str(i+1) + ': starting')
    
            #train = folds[i][0]+folds[i][2]
            train = folds[i][0]   # original qdo nao utiliza o background
            for j in range(n_folds):
                if j != i:
                    train = train + folds[j][0]+folds[j][1]#+folds[j][2]
    
            t_start = time.time()
            LOGGER.info('le_descritores_train: starting')
            ds, id_ds = ls.le_descritores(sift_folder, sift_type, train, 2)
            LOGGER.info('le_descritores_train: ending(' + str(time.time()-t_start)+')')
    
    #%%
    
            # gera o codebook com apenas a base reduzida
            k = bag_size
            t_start = time.time()
            LOGGER.info('bov_codebook_gera: starting')
            centers, labels = ls.bov_codebook_gera(ds, k, 2)
            # tipo = 1 -> Kmeans, tipo=2 ->minibatch, tipo=3 -> random
            LOGGER.info('bov_codebook_gera: ending(' + str(time.time()-t_start)+')')
    
    #%%
            #path = "/media/agnus/My Passport/Projeto/dat_artigo/"
    
            t_start = time.time()
            LOGGER.info('bov_codebook_grava: starting')
            ls.bov_codebook_grava(path + '/BOV_codebook_'+str(i+1)+'_'+DT+".txt", centers, DT)
            LOGGER.info('bov_codebook_grava: ending(' + str(time.time()-t_start)+')')
    
    #%%
            # alteração para permitir que considere toda a base
            if len(subsets)>2 :
                train = train+folds[i][2]
    
                if i==0: # Read the baground data only in the first time
                    t_start = time.time()
                    LOGGER.info('le_descritores_train expandido: starting')
                    ds1, id_ds1 = ls.le_descritores(sift_folder, sift_type, folds[i][2], 2)
                    LOGGER.info('le_descritores_train expandido: ending(' + str(time.time()-t_start)+')')
    
                ds = np.concatenate((ds, ds1), axis=0)
                id_ds.extend(id_ds1)
    
    #%%
            #import random
            from scipy.cluster.vq import vq
    
            t_start = time.time()
            LOGGER.info('rotula_descritores e calcula distancia ao centroids: starting')
    
            fim = False
            start_i = 0
    
            while not fim:
                end_i = start_i+5000
    
                if end_i > len(ds):
                    end_i = len(ds)
                    fim = True
    
                lbl, dst_cen = vq(ds[start_i:end_i], centers)
    
                if start_i == 0:
                    labels = np.empty_like(lbl)
                    labels[:] = lbl
                    dist_cen = np.empty_like(dst_cen)
                    dist_cen[:] = dst_cen
    
                else:
                    labels = np.concatenate((labels, lbl), axis=0)
                    dist_cen = np.concatenate((dist_cen, dst_cen), axis=0)
    
                start_i = end_i # +1,  removed due to the way the slice works in Python
    
            LOGGER.info('rotula_descritores e calcula distancia ao centroids: ending(' + str(time.time()-t_start)+')')
    #%%
    
            t_start = time.time()
            LOGGER.info('bov_histogramas_gera: starting')
            hists_train = ls.bov_histogramas_gera(labels, dist_cen, id_ds, k, train, vis=False, th=1000)
            LOGGER.info('bov_histogramas_gera: ending(' + str(time.time()-t_start)+')')
    
    #%%
            t_start = time.time()
            LOGGER.info('bov_histogramas_grava train: starting')
            ls.bov_histogramas_grava(path + 'BOV_hists_train_'+str(i+1)+'_'+DT+".txt", hists_train, DT)
            LOGGER.info('bov_histogramas_grava train: ending(' + str(time.time()-t_start)+')')
    
    #%%
    
            classes = []
            for image in train:
                if class_gallery.has_key(image):
                    classes.append(class_gallery[image])
                else:
                    classes.append(class_probe[image])
    
            ls.gera_svm_dat(i, classes, hists_train, DT, False)
    
    #%%
            # Inicialmente esta considerando apenas um fold, deve ser verifcado o caso
            # de ter mais de um fold
    
            #n_folds = len(folds) # por enquanto n_folds será 1 pois tem apenas um fold
            #for i in range(n_folds):
            test = folds[i][1]
    
            #ds = ls.le_descritores(sift_folder, test)
            t_start = time.time()
            LOGGER.info('le_descritores_test: starting')
            ds, id_ds = ls.le_descritores(sift_folder, sift_type, test, 2)
            LOGGER.info('le_descritores_test: ending(' + str(time.time()-t_start)+')')
    
            t_start = time.time()
            LOGGER.info('bov_descritores_codifica test: starting')
            labels = ls.bov_descritores_codifica(ds, centers)
            LOGGER.info('bov_descritores_codifica test: ending(' + str(time.time()-t_start)+')')
    
    #%%
    
            t_start = time.time()
            LOGGER.info('bov_histogramas_gera test: starting')
            hists_test = ls.bov_histogramas_gera(labels, dist_cen, id_ds, k, test, vis=False)
            LOGGER.info('bov_histogramas_gera test: ending(' + str(time.time()-t_start)+')')
    #%%
            t_start = time.time()
            LOGGER.info('bov_histogramas_grava test: starting')
            ls.bov_histogramas_grava(path + 'BOV_hists_test_'+str(i+1)+'_'+DT+".txt", hists_test, DT)
            LOGGER.info('bov_histogramas_grava test: ending(' + str(time.time()-t_start)+')')
    
    #%%
            classes = []
            for image in test:
                classes.append(class_probe[image])
    
    
            ls.gera_svm_dat(i, classes, hists_test, DT, True)
    
    #%%
            import scipy.spatial.distance as ssd
    
            # não está considerando a questão de multiplos folds
            # e também o uso de memória auxiliar no disco
    
            ntrain = len(hists_train)
    
            #i = 0
            arquivo = path + 'clist_mem_'+str(i+1)+'.txt'
            with open(arquivo, 'w') as clist_file:
    
                ntest = len(hists_test)
    
                for i_test in range(ntest):
    
                    file_test = test[i_test]
                    u = hists_test[i_test]
                    dist = np.zeros((ntrain))
    
                    for i_train in range(ntrain):
    
                        v = hists_train[i_train]
                        #dist[i_train] = ssd.cityblock(u, v)
                        dist[i_train] = ssd.euclidean(u, v)
    
                    #indice = np.argsort(dist)[::-1]
                    indice = np.argsort(dist)
    
                    k = 1
                    for idx in indice:
                        clist_file.write(file_test+'|'+ str(k) +
                                         '|' + train[idx] + '|' + str(dist[idx]) +'\n')
                        k = k + 1
    
            #gc.collect()
            
            LOGGER.info('processa_' + metodo + ' Folder : ending(' + str(time.time()-t_start_folder)+')')
    
    #%%
    #i = 0
    #arquivo = './clist_mem_'+str(i+1)+'.txt'
    t_start = time.time()
    LOGGER.info('ground_truth: starting')
    cmc = ls.compute_cmc(gt_imagens, path, metodo)
    LOGGER.info('ground_truth: ending(' + str(time.time()-t_start)+')')
    
    #%%
    ls.plot_cmc(cmc, metodo, path + "cmc_"+metodo+".png", nf=100, dx=10 )
    
    #%%
    os.rename(path[:-1], path[:-1] + "_" + metodo)
    
    if not bag is None:
        if "BOV" in bag:
            #%%
            print (min(dist_cen))
            print (max(dist_cen))
            
            #%%
            import matplotlib.pyplot as plt
            plt.hist(dist_cen, bins=40)
            plt.show()

    HANDLER.close()
    LOGGER.removeHandler(HANDLER)
    
#%%
    
if __name__ == "__main__":

    #metodos = ['a_kaze_050', 'aakaze_050', 'afreak_050', 'abrief_050', 'abrisk_050', 'aorb_050', 'asurf_050', 'asift_050']
    metodos = ['sift']
    n_subsets = 3 #number of subsets, 3 for include background, 2 otherwise
    bg_perc = 100 #The percentage of background usage, for example 25, indicates that only 25% of all background images will be used.
                #must be use only if n_subsets>2
                
    for metodo in metodos:
        
        print(metodo)
        #bags=['BOV', 'FV', None]
        bags=['BOV','FV']
        
        for bag in bags:
        
            if not bag is None:
                if 'FV' in bag:
                    #sizes=[1,2,3,4,5,10,15]
                    sizes=[5]
                elif 'BOV' in bag:
                    #sizes=[10, 25, 50, 100, 200, 500, 1000]
                    sizes=[50]
            else:
                sizes=[0]
                
            for size in sizes:
            
                if bag != None:
                    filename = bag + '_'
                else:
                    filename = ""
                    
                fname = filename + metodo
                if size!=0:
                    fname = fname  + '_' + str(size).zfill(5)
                
                filename = filename + metodo + '_' + str(size).zfill(5)
                
                #if bg_perc>0:
                filename = filename + '_BG' + str(bg_perc).zfill(3) 
                fname = fname + '_BG' + str(bg_perc).zfill(3)
                
                print (fname)
                
                if not os.path.exists("/Projeto/Projetos/dat_artigo_" + fname.upper()):
            
                    t_start_m = time.time()
                    
                    mf = open("/Projeto/Projetos/master.log" ,'a')
                    mf.write((datetime.now()).strftime("%Y-%m-%d %H:%M:%S") + ":" +  filename  + ": starting \n" )
                    mf.close()
                    
                    rodada (filename, metodo, bag, size, n_subsets, bg_perc)
                    
                    mf = open("/Projeto/Projetos/master.log" ,'a')
                    mf.write((datetime.now()).strftime("%Y-%m-%d %H:%M:%S") + ":" + filename + ": ending (" + str(time.time()-t_start_m) + ")\n" )
                    mf.close()
                    
                else:
                    print("Folder exist!")
