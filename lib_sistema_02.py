# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:36:05 2016

@author: agnus
"""
#%%
def monta_lista_imagens(path = '.', ext='.png'):
    import os

    print (path, ext)

    imagens = {}
    for dirname, dirnames, filenames in os.walk(path):
        # print path to all filenames with extension py.
        if "tattoo_identification" in dirname: # or ("background" in dirname and "orig" in dirname):
            for filename in filenames:
                fname_path = os.path.join(dirname, filename)
                fext = os.path.splitext(fname_path)[1]
                if fext == ext:
                    #file_dat = [filename, dirname]
                    #imagens.append(file_dat)
                    imagens[filename]=dirname
                else:
                    continue

    return imagens

#%%
def grava_db_imagens(arquivo, imagens):
    #arquivo = './tatt_c.db'
    with open(arquivo, 'wb') as db_image_file:
        for  nome_img, caminho in imagens.items():
            db_image_file.write(nome_img+ '\t' + caminho + '\n')
        db_image_file.close()

#%%

def grava_config(arquivo, metodo, bag, size, n_subsets, bg):
#def grava_config(arquivo = '/home/agnus/Documentos/Projeto/dat_artigo/example_mem.cfg'):

    import configparser as ConfigParser

    config = ConfigParser.RawConfigParser()

    # When adding sections or items, add them in the reverse order of
    # how you want them to be displayed in the actual file.
    # In addition, please note that using RawConfigParser's and the raw
    # mode of ConfigParser's respective set functions, you can assign
    # non-string values to keys internally, but will receive an error
    # when attempting to write to a file or when you get it in non-raw
    # mode. SafeConfigParser does not allow such assignments to take place.

    config.add_section('Geral')
    config.set('Geral', 'Image Database', 'Tatt-C')

    config.set('Geral', 'Database Image Folder', '/Projeto/dataset/tatt-c/')

    config.set('Geral', 'Indexa image database', 'True')

    config.set('Geral', 'Database filename', '/Projeto/Projetos/dat_artigo/tatt_c.db')

    config.set('Geral', 'Image filename extension','.jpg')

    config.set('Geral', 'Training File', 'train1')
    config.set('Geral', 'Testing File', 'test1')

    config.add_section('Folds')

    config.set('Folds', 'Folds Folder', '/Projeto/dataset/tatt-c_update_v1.4/5-fold/tattoo_identification/')
    config.set('Folds', 'Quantidade subsets', n_subsets)
    config.set('Folds', 'Subset_1', 'gallery{1,2,3,4,5}.txt')
    config.set('Folds', 'Subset_2', 'probes{1,2,3,4,5}.txt')
    
    if n_subsets==3:
        config.set('Folds', 'Subset_3', 'bg{1,1,1,1,1}.txt')
        config.set('Folds', 'BG', bg)
    
    config.set('Folds', 'Ground_truth', 'ground_truth_classes.txt')

    config.add_section('SIFT')
    config.set('SIFT','SIFT Folder',  '/Projeto/dataset/descriptors/')
 
    config.set('SIFT','SIFT Type',  metodo)

    config.add_section('BAG') # FV
    config.set('BAG','BAG Type', bag)
    config.set('BAG','BAG Size', size)

    # Writing our configuration file to 'example.cfg'
    with open(arquivo, 'w') as configfile:
        config.write(configfile)


#%%
def folds_construct(subsets, folds_folder, bg_perc):

    import random
    
    n_folds =len(subsets[0])
    n_subsets = len(subsets)
 
    folds = []
    images2exclude = []
    
    #le a lista de imagens a serem desconsideradas
    with open("/Projeto/dataset/tatt-c/"+"files2exclude.txt", 'r') as imagefiles:
        for nomef in imagefiles:
            if nomef[-1] == '\n' : nomef = nomef[:-1]
            images2exclude.append(nomef)
    imagefiles.close()
            
    for i in range(n_folds):
 
        random.seed(30) #to generate the same sequence of numbers every time
        
        sub = []
        for j in range(n_subsets):
            
            arquivo = subsets[j][i]
        
            aux = []
            with open(folds_folder+arquivo, 'r') as imagefiles:
                for nomef in imagefiles:
                    if nomef[-1] == '\n' : nomef = nomef[:-1]
                    index = nomef.rfind('/')
                    if not (nomef[index+1:] in images2exclude):
                        aux.append(nomef)
            imagefiles.close()
            
            if j==2:
                k = int(bg_perc*len(aux)/100)
                aux1 = random.sample(aux,k)
                sub.append(aux1)
            else:    
                sub.append(aux)
        
        folds.append(sub)

    return folds

#%%
def le_config(filename):
    import configparser as ConfigParser

    config = ConfigParser.RawConfigParser()
    config.read(filename)
 

    # getfloat() raises an exception if the value is not a float
    # getint() and getboolean() also do this for their respective types
    base = config.get('Geral', 'image database')
    indexa = config.getboolean('Geral', 'indexa image database')
    print (base)
    if indexa:
        print ("indexa base")
        arquivo = config.get('Geral','database filename')
        caminho = config.get('Geral', 'database image folder')
        extensao = config.get('Geral', 'image filename extension')

        print (arquivo, caminho, extensao)
        imagens = monta_lista_imagens(caminho, extensao)

        #grava_db_imagens(arquivo, imagens)

    folds_folder = config.get('Folds','folds folder')
    
    if "/media" in folds_folder[:6]:
        folds_folder = '/' + folds_folder[10:]
    
    n_subsets = config.getint('Folds', 'quantidade subsets')

    subsets=[]

    for i in range(n_subsets):
        sub = config.get('Folds', 'subset_'+str(i+1))
        ps = sub.find("{")
        pe = sub.find("}")
        ped = sub[ps+1:pe]
        indices = ped.split(',')
        aux = []
        for ind in indices:
            aux.append(sub[:ps]+ind+'.txt') # incluir extensão variável
        subsets.append(aux)

    print (subsets)

    #n_folds = config.getint('Folds', 'quantidade folds')
    n_folds =len(subsets[0])
    folds = []
    for i in range(n_folds):
        sub = []
        for j in range(n_subsets):
            print (i, j)
            arquivo = subsets[j][i]
            aux = []
            with open(folds_folder+arquivo, 'r') as imagefiles:
                for nomef in imagefiles:
                    if nomef[-1] == '\n' : nomef = nomef[:-1]
                    aux.append(nomef)
            imagefiles.close()
            sub.append(aux)
        folds.append(sub)

    #print folds[0]

    if n_subsets==3:
        bg = config.getint('Folds', 'BG')
    else:
        bg = 0
        
    gt_filename = config.get('Folds', 'ground_truth')

    sift_folder = config.get('SIFT', 'sift folder')
    sift_type = config.get('SIFT', 'sift type')

    #bov_K = int(config.get('BOV','Codebook Size'))

    print (sift_folder, folds_folder, caminho)

    bag_type = config.get('BAG','BAG Type')
    bag_size = int(config.get('BAG','BAG Size'))

    return (folds, imagens, gt_filename, sift_folder, sift_type, folds_folder, caminho, subsets, bag_type, bag_size, bg)

#%%
def sift(nomes_imagens, imagens, sift_folder, sift_type):

    import cv2
    import os
    from math import sqrt

    #ds = []
    #kp = []
    t = len(nomes_imagens)
    i=1


    for filename in nomes_imagens:

        fname = os.path.join(sift_folder, filename[:-3] + sift_type + '_ds')

        if os.path.isfile(fname) == False :
            print (filename)
            #file_img = os.path.join(diretorio, filename)
            diretorio = imagens[filename]
            img = cv2.imread(os.path.join(diretorio, filename)) #file_img)
            # Redimensiona imagem para aplicação do Fisher Vectors
            #img = cv2.resize(img, (256,256))
            aux = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(aux)
            k = sqrt((240.0*480.0*1.0)/(gray.shape[0]*gray.shape[1]))
            #k=1
            res = cv2.resize(gray,None,fx=k, fy=k, interpolation = cv2.INTER_CUBIC)
            cv2.imwrite("/media/sf_Projeto/dataset/tatt-c_artigo_v1/img_tratada/"+filename,res)
            sift = cv2.xfeatures2d.SIFT_create()
            (kps, descs) = sift.detectAndCompute(res, None)

            #ds.append(descs)
            #kp.append(kps)

            arquivo = os.path.join(sift_folder, filename[:-3] + sift_type + '_ds')
            with open(arquivo, 'wb') as sift_file:
                for desc in descs:
                    sift_file.write(','.join(str(x) for x in desc)+'\n')
                sift_file.close()

            arquivo = os.path.join(sift_folder, filename[:-3] + sift_type + '_kp')
            with open(arquivo, 'wb') as sift_file:
                for point in kps:
                    temp = [point.pt[0], point.pt[1], point.size, point.angle,
                        point.response, point.octave, point.class_id]
                    sift_file.write(','.join(str(x) for x in temp)+'\n')
                sift_file.close()

        print (i*100)/t,
        i=i+1

    #return ds

#%%
def sift_match(ds1, kp1, ds2, kp2):

    import cv2
    import numpy as np

    if 1==1:

        #MIN_MATCH_COUNT = 10

        bf = cv2.BFMatcher()

        #print ds1, ds2

        #matches = bf.knnMatch(ds1,ds2, k=2)
        matches = bf.knnMatch(np.asarray(ds1,np.float32),np.asarray(ds2,np.float32),k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        qm = len(good)

    else:

        qm = 0
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(ds1,ds2,k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in xrange(len(matches))]

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
                qm = qm + 1


    #print type(ds1), ds1
    #print type(ds2), ds2
    
    
    (nr1,c) = ds1.shape
    (nr2,c) = ds2.shape

    #    if qm>MIN_MATCH_COUNT:
    #        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #
    #        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #        if mask != None:
    #            matchesMask = mask.ravel().tolist()
    #            rt = np.sum(np.asarray(matchesMask))
    #        else:
    #            rt = 0
    #    else:
    #        #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    #        #matchesMask = None
    #        rt = 0

    nr = nr1
    if nr2>nr:
        nr = nr2

    rt = (100.0*qm/nr)
    #    if qm > 0:
    #        rt = 1.0/qm
    #    else:
    #        rt = 10^8

    return rt

#%%
def gera_sift_base(folds, imagens, sift_folder, sift_type):

    # Inicialmente gera se necessario o SIFT para as imagens de treinamento e teste
    # pode ser otimizado, gerando para toda a base, caso se utilize toda a base
    # o que pode ter um custo alto pois na base existem imagens para outros casos
    # de uso.
    n_folds = len(folds)
    #Poder ser implementado diferente pois as linhas abaixo apenas agregram os nomes
    #das imagens para que sejam gerados os sifts para cada um dos folds
    for i in range(n_folds):
        test = folds[i][1]

        if len(folds[i])>2 :
            train = folds[i][0]+folds[i][2] # retirar folds[i][2] qdo não utilizar background
        else:
            train = folds[i][0]

        #bg = folds[i][2]
        for j in range(n_folds):
            if j!=i :
                train = train + folds[j][0]+folds[j][1] #+folds[j][2]

        print ('Gerando sift do conjunto de treinamento')
        #train_kp, train_ds = sift(train, imagens, sift_folder)
        sift(train, imagens, sift_folder, sift_type)
        print ('Gerando sift do conjunto de teste')
        #test_kp, test_ds = sift(test, imagens)
        sift(test, imagens, sift_folder, sift_type)
        #print 'Gerando sift do conjunto de bg'
        ##bg_kp, bg_ds = sift(bg, imagens)
        #sift(bg, imagens, sift_folder, sift_type)

#%%
def processa_sift(folds, imagens, sift_folder, sift_type, LOGGER, metodo, subsets):

    import numpy as np
    import os
    import cv2
    import time


    n_folds = len(folds)
    
    if len(subsets)>2:
        n_bg = len(folds[0][2])
    else:
        n_bg = 0
        
    #Alterei para que inclua nas imagens da galeria i no conj. train, de forma a que as
    # imagens correspondentes ao probe existam na galeria (train)
    for i in range(n_folds):
        
        t_start = time.time()
        LOGGER.info('processa_' + metodo + ' Folder ' + str(i+1) + ': starting')
        test = folds[i][1]
        
        if len(subsets)>2:
            train = folds[i][2] + folds[i][0] # background + train subset of the fold
        else:
            train = folds[i][0]
            
        for j in range(n_folds):
            if j!=i :
                train = train + folds[j][0]+folds[j][1]

        n_test = len(test)
        n_train = len(train)

        dist = np.zeros((n_train), dtype=np.float)
        nn = n_test * n_train

        print ('Gerando o match entre o treinamento e o conjunto de teste')

        mem = True
        if mem==True and i == 0 :
            ds=[]
            ks=[]

        arquivo = '/Projeto/Projetos/dat_artigo/clist_mem_'+str(i+1)+'.txt'
        #arquivo = '/media/agnus/My Passport/Projeto/dat_artigo/clist_mem_'+str(i+1)+'.txt'
        

        with open(arquivo, 'w') as clist_file:

            l = 0

            for file_test in test:

                mask_folder = '/Projeto/dataset/masks/'
                #print file_test
                
                mask = mask_read(mask_folder, file_test)
        
                kp1_aux, kp_type, kp_num = kp_read(sift_folder,file_test, sift_type)
                aux, _, _, _ = ds_read(sift_folder, file_test, sift_type)

                #print len(kp1), len(aux)
                
                ds1_aux = []
                kp1 = []
                
                for kpt, des in zip(kp1_aux,aux):
                    if mask[int(kpt.pt[1]), int(kpt.pt[0])] == 1:
                        ds1_aux.append(des)
                        kp1.append(kpt)
                        
                ds1 = np.asarray(ds1_aux, dtype=np.float32)
                #ds1 = ds1_aux
                
                # as linhas abaixo foram comentadas em função do uso da mascara       
        
                #ds1, _, _, _ = ds_read(sift_folder, file_test, sift_type)

                #if sift_type == 'mhlscd':
                #    fname = os.path.join(sift_folder, file_test[:-3] + 'sift' + '_kp')
                #else:
                #    fname = os.path.join(sift_folder, file_test[:-3] + sift_type + '_kp')
                    
                ##print ("==> ", file_test)
                #kp1, _, _ = kp_read(sift_folder, file_test, sift_type)

                ##diretorio = imagens[file_test]
                ##img1 = cv2.imread(os.path.join(diretorio, file_test),0)
                ##print os.path.join(diretorio, file_test)
                j = 0

                for file_train in train:
                    #diretorio = imagens[file_train]
                    #img2 = cv2.imread(os.path.join(diretorio, file_train),0)
                    #print os.path.join(diretorio, file_train)

                    print(i, l, j, n_bg, len(ds))

                    if (mem == True and ( i == 0 and l ==0) or (i > 0 and l == 0 and j >= n_bg)): #(mem == True and len(ds)<len(train)):
#                        fname = os.path.join(sift_folder, file_train[:-3] + sift_type + '_ds')
#                        # mudança do tipo de uint8 para float em fução do mhlscd 03/10/2016
#                        aux = np.asarray((np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.float)) #* 255
#                        ds.append ( aux.astype(np.float32)  ) #,skiprows=1)
#                        #print fname
#                        ds2 = ds[j]

                        #print (file_train)

                        mask_folder = '/Projeto/dataset/masks/'
                        mask = mask_read(mask_folder, file_train)
        
                        kp2_aux, kp_type, kp_num = kp_read(sift_folder,file_train, sift_type)
                        aux, _, _, _ = ds_read(sift_folder, file_train, sift_type)

                        ds2_aux = []
                        kp2 = []
                        
                        for kpt, des in zip(kp2_aux,aux):
                            if mask[int(kpt.pt[1]), int(kpt.pt[0])] == 1:
                                ds2_aux.append(des)
                                kp2.append(kpt)

                        ds2 = np.asarray(ds2_aux, dtype=np.float32)
                
                        #ds2, _, _, _ = ds_read(sift_folder, file_train, sift_type)
                        ds.append(ds2)

                        fname = os.path.join(sift_folder, file_test[:-3] + sift_type + '_kp')

                        #kp2, _, _ = kp_read(sift_folder, file_train, sift_type)
                        ks.append(kp2)

                    elif (mem == True): # and len(ds)==len(train)):
                        ds2 = ds[j]
                        kp2 = ks[j]
                    elif mem == False:
                        fname = os.path.join(sift_folder, file_train[:-3] + sift_type + '_ds')
                        # mudança do tipo de uint8 para float em fução do mhlscd 03/10/2016
                        ds2 = ( (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.float) )

                        fname = os.path.join(sift_folder, file_train[:-3] + sift_type + '_kp')
                        kps = (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.float) #,skiprows=1)

                        kp2 = []
                        for kp in kps:
                            kpoint = cv2.KeyPoint(float(kp[0]), float(kp[1]),
                                      float(kp[2]), float(kp[3]),
                                      float(kp[4]), int(kp[5]), int(kp[6]))
                            kp2.append(kpoint)

                    #print ds1
                    #print ds2
                    rt = sift_match(ds1, np.asarray(kp1), ds2, np.asarray(kp2))
                    dist[j] = rt
                    j = j + 1
                    print (i,(((l*n_train)+j)*100)/nn)

#                if n_bg>0:
#                    ds = ds[:n_bg]
#                    ks = ks[:n_bg]
                    
                indice = np.argsort(dist)[::-1]
                k = 1
                for id in indice:
                    clist_file.write(file_test+'|'+ str(k) + '|' + train[id] + '|' + str(dist[id]) +'\n')
                    k = k + 1

                l = l + 1

            clist_file.close()

            # When the current folder is ready, in the next folder, delete ds and ks data for non-background images.
            if n_bg>0:
                ds = ds[:n_bg]
                ks = ks[:n_bg]
            else:
                ds = []
                ks = []

            LOGGER.info('processa_' + metodo + ' Folder : ending(' + str(time.time()-t_start)+')')

#%%
def kp_read(sift_folder, filename, sift_type):

    import csv
    #import numpy as np
    import os
    import cv2

    kp_type = ''
    kp_num = 0

    fname = os.path.join(sift_folder, filename[:-4] + '_' + sift_type + '_kp.csv') #versao atual
    #fname = os.path.join(sift_folder, filename[:-4] + '.' + sift_type + '_kp') # versao wscg

    with open(fname, 'r') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        rownumber = 0

        kp = []

        #kp_num = spamreader.line_num

        for row in spamreader:

            if rownumber == 0:

                kp_type = row[0]
                kp_num = row[1]
                rownumber = rownumber + 1

            else:

                kpoint = cv2.KeyPoint(float(row[0]), float(row[1]),
                                  float(row[2]), float(row[3]),
                                  float(row[4]), int(row[5]), int(row[6]))
                kp.append(kpoint)
                rownumber = rownumber + 1

            #if rownumber>25000:
            #    break

        return kp, kp_type, kp_num


#%%
def ds_read(sift_folder, filename, sift_type):

    import csv
    import numpy as np
    import os

    ds_type = ''
    ds_num = 0
    ds_dim = 0

    fname = os.path.join(sift_folder, filename[:-4] + '_' + sift_type + '_des.csv') # versao atual
    #fname = os.path.join(sift_folder, filename[:-4] + '.' + sift_type + '_ds') # versao wscg


    with open(fname, 'r') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        rownumber = 0

        aux = []

        #ds_num = spamreader.line_num

        for row in spamreader:

            #print row

            if rownumber == 0:

                ds_type = row[0]
                ds_num = row[1]
                ds_dim = row[2]
                rownumber = rownumber + 1

            else:
                #print row
                #row = map(int, row) # needed due wscg data file
                aux.append(row)
                rownumber = rownumber + 1

            #if rownumber>25000:
            #    break

        #ds = np.asarray(aux, dtype=np.float32)
        ds = aux
   
        return ds, ds_type, ds_num, ds_dim


#%%
def processa_surf(folds, imagens, sift_folder, sift_type, LOGGER):

    import numpy as np
    import os
    import cv2
    import time

    n_folds = len(folds)
    #Alterei para que inclua nas imagens da galeria i no conj. train, de forma a que as
    # imagens correspondentes ao probe existam na galeria (train)
    for i in range(n_folds):
        t_start = time.time()
        LOGGER.info('processa_surf Folder ' + str(i+1) + ': starting')
        test = folds[i][1]
        #bg = folds[i][2]
        train = folds[i][0]#+bg
        for j in range(n_folds):
            if j!=i :
                train = train + folds[j][0]+folds[j][1]#+folds[j][2]

        n_test = len(test)
        n_train = len(train)

        dist = np.zeros((n_train), dtype=np.float)
        nn = n_test * n_train

        print ('Gerando o match entre o treinamento e o conjunto de teste')

        mem = True
        if mem==True :
            ds=[]
            ks=[]

        arquivo = '/Projeto/Projetos/dat_artigo/clist_mem_'+str(i+1)+'.txt'

        with open(arquivo, 'w') as clist_file:

            l = 0

            for file_test in test:

                ds1, _, _, _ = ds_read(sift_folder, file_test, sift_type)

                if sift_type == 'mhlscd':
                    fname = os.path.join(sift_folder, file_test[:-3] + 'sift' + '_kp')
                else:
                    fname = os.path.join(sift_folder, file_test[:-3] + sift_type + '_kp')

                kp1, _, _ = kp_read(sift_folder, file_test, sift_type)

                diretorio = imagens[file_test]
                img1 = cv2.imread(os.path.join(diretorio, file_test),0)
                #print os.path.join(diretorio, file_test)
                j = 0

                for file_train in train:
                    diretorio = imagens[file_train]
                    img2 = cv2.imread(os.path.join(diretorio, file_train),0)
                    #print os.path.join(diretorio, file_train)
                    if (mem == True and len(ds)<len(train)):

                        ds2, _, _, _ = ds_read(sift_folder, file_train, sift_type)
                        ds.append(ds2)

                        #fname = os.path.join(sift_folder, file_train[:-3] + sift_type + '_kp')
                        if sift_type == 'mhlscd':
                            fname = os.path.join(sift_folder, file_test[:-3] + 'sift' + '_kp')
                        else:
                            fname = os.path.join(sift_folder, file_test[:-3] + sift_type + '_kp')

                        kp2, _, _ = kp_read(sift_folder, file_train, sift_type)
                        ks.append(kp2)

                    elif (mem == True and len(ds)==len(train)):
                        ds2 = ds[j]
                        kp2 = ks[j]
                    elif mem == False:
                        fname = os.path.join(sift_folder, file_train[:-3] + sift_type + '_ds')
                        # mudança do tipo de uint8 para float em fução do mhlscd 03/10/2016
                        ds2 = ( (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.float) )

                        fname = os.path.join(sift_folder, file_train[:-3] + sift_type + '_kp')
                        kps = (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.float) #,skiprows=1)

                        kp2 = []
                        for kp in kps:
                            kpoint = cv2.KeyPoint(float(kp[0]), float(kp[1]),
                                      float(kp[2]), float(kp[3]),
                                      float(kp[4]), int(kp[5]), int(kp[6]))
                            kp2.append(kpoint)

                    #print ds1
                    #print ds2
                    rt = sift_match(ds1, np.asarray(kp1), ds2, np.asarray(kp2))
                    dist[j] = rt
                    j = j + 1
                    print (i,(((l*n_train)+j)*100)/nn)

                indice = np.argsort(dist)[::-1]
                k = 1
                for id in indice:
                    clist_file.write(file_test+'|'+ str(k) + '|' + train[id] + '|' + str(dist[id]) +'\n')
                    k = k + 1

                l = l + 1

            clist_file.close()

            LOGGER.info('processa_surf Folder : ending(' + str(time.time()-t_start)+')')


#%%

def ground_truth(folds_folder, gt_filename):
    """Reads a ground truth table from text file.

    Keyword arguments:
    folds_folder -- the path for the ground truth file
    gt_filename -- the file name of the ground truth file with extension

    Returns:
    gt_images -- ground truth table stored in a dictionary
    """

    #folds_folder = '/media/sf_Projeto/dataset/tatt-c_update_v1.4/5-fold/tattoo_identification/'
    #gt_filename = 'ground_truth.txt'
    gt_imagens = {}
    class_probe = {}
    class_gallery = {}

    with open(folds_folder+gt_filename, 'r') as gt_arq:
        for nomef in gt_arq:
            imgs = nomef.split('|')
            if imgs[2][-1] == '\n' : imgs[2] = imgs[2][:-1]
            #print imgs[0], imgs[1]

            #if gt_imagens.has_key(imgs[0]):
            if imgs[0] in gt_imagens:
                gt_imagens[imgs[0]].append(imgs[1])
            else:
                gt_imagens[imgs[0]] = [imgs[1]]

            #if not class_probe.has_key(imgs[0]):
            if imgs[0] not in class_probe:    
                class_probe[imgs[0]] = imgs[2]

            #if not class_gallery.has_key(imgs[1]):
            if imgs[1] not in class_gallery:    
                class_gallery[imgs[1]] = imgs[2]

        gt_arq.close()

    return gt_imagens, class_probe, class_gallery

#%%
def compute_cmc(gt_imagens, path = '/Projeto/Projetos/dat_artigo/', metodo="SIFT"):
    """Reads a classification list from text file and sumarize rank results for
        every image reference based in the ground truth dictionary.

    Keyword arguments:
    arquivo -- the filename of classification list file
    gt_images -- ground truth table stored in a dictionary

    Returns:
    cmc -- acummulated accuracy for each rank stored in a numpy array
    """
    import numpy as np

    acc = np.zeros(5000)
    for i in range(5):
        arquivo = path + 'clist_mem_' + str(i+1) + '.txt'
        with open(arquivo, 'r') as clist_file:
            for nomef in clist_file:
                imgs = nomef.split('|')
                if imgs[3][-1] == '\n' : imgs[3] = imgs[3][:-1]
                if imgs[2] in gt_imagens[imgs[0]]:
                    r = int(imgs[1])
                    acc[r] = acc[r]+1
        clist_file.close()

    acc = acc*0.2
    #print cmc
    ft = sum(acc)
    #print cmc/ft
    cmc = np.zeros(5000)
    for i in range(1,5000):
        cmc[i] = cmc[i-1]+acc[i]/ft
    #print cmc1

    #print path+"cmc_" + metodo + ".dat"
    np.savetxt(path+"cmc_" + metodo + ".dat", cmc)

    return cmc

#%%
def plot_cmc(cmc, metodo, fl, nf=50, dx=300, w = 10, h = 7 ):

    import matplotlib.pyplot as plt
    import pylab as P
    import numpy as np

    fig = P.figure()
    fig.suptitle('Cumulative Match Characteristic', fontsize=18, fontweight='bold')
    P.title(metodo)

    P.ylabel('%', fontsize=16)
    P.xlabel('Rank', fontsize=16)

    P.xlim(0, nf)
    P.ylim(0,101)
    P.xticks(np.arange(0, nf, dx))
    P.yticks(np.arange(0, 101, 5.0))

    xticklabels = P.getp(P.gca(), 'xticklabels')
    yticklabels = P.getp(P.gca(), 'yticklabels')

    P.setp(yticklabels, 'color', 'k', fontsize='x-large')
    P.setp(xticklabels, 'color', 'k', fontsize='x-large')

    P.grid(True)
    fig.set_size_inches(w, h)
    #P.plot(cmc*100)
    P.plot(cmc*100)
    fig.savefig(fl)
    P.show()

#%%%

#Author: Jacob Gildenblat, 2014
#http://jacobcv.blogspot.com.br/2014/12/fisher-vector-in-python.html
#License: you may use this for whatever you like
#Adaptation: Agnus A. Horta

def fv_dictionary(descriptors, N):

    import numpy as np
    import cv2

    em = cv2.ml.EM_create()
    em.setClustersNumber(N)
    #em = cv2.EM(N)
    em.trainEM(descriptors)

    return np.float32(em.getMeans()), \
        np.float32(em.getCovs()), np.float32(em.getWeights())[0]

def fv_generate_gmm(i, descriptors, N, dt):

    #from vlfeat.gmm import *
    import vlfeat.gmm as vg
    #from vlfeat.fisher import * 
    import vlfeat.fisher as vf
    from random import random
    import numpy as np

    words = np.concatenate(descriptors)

#%%

    array_data = np.concatenate(descriptors)
    
    sigmaLowerBound = 0.000001
    
    (numData, dimension) = array_data.shape
    numClusters = N
    maxiter = 5
    maxrep = 1
    
#%%
    
    # create a new instance of a GMM object for float data
    gmm = vg.vl_gmm_new (vg.VL_TYPE_FLOAT, dimension, numClusters)
    
#%%
    #set the initialization to random selection
    vg.vl_gmm_set_initialization(gmm,vg.VlGMMRand)
    
#%%
    vg.vl_gmm_set_max_num_iterations(gmm, maxiter)
    vg.vl_gmm_set_num_repetitions(gmm, maxrep)
    vg.vl_gmm_set_verbosity(gmm,0)
    #vl_gmm_set_covariance_lower_bound (gmm,sigmaLowerBound)

#%%
   
    #array_data = np.concatenate(descriptors)

#%%
    
    vg.vl_gmm_cluster(gmm, array_data, numData)
    
#%%
    
    means = vg.vl_gmm_get_means(gmm)
    covs = vg.vl_gmm_get_covariances(gmm)
    weights = vg.vl_gmm_get_priors(gmm)
    
#%%
    #path = "/media/agnus/My Passport/Projeto/dat_artigo/"
    path = "/Projeto/Projetos/dat_artigo/"

    np.save(path + "means_" + str(i) + '_' + dt + ".gmm", means)
    np.save(path + "covs_" + str(i) + '_' + dt + ".gmm", covs)
    np.save(path + "weights_" + str(i) + '_' + dt + ".gmm", weights)

    return means, covs, weights

def fv_load_gmm(i, dt, folder = "/Projeto/Projetos/dat_artigo"):

    import numpy as np

    files = ["means_" + str(i) + '_' + dt + ".gmm" +".npy", "covs_" + str(i) + '_' + dt + ".gmm.npy", "weights_" + str(i) + '_' + dt + ".gmm.npy"]

    try:
        return map(lambda file: np.load(file), map(lambda s : folder + "/" + s , files))

    except IOError:
        return (None, None, None)

def fv_likelihood_moment(x, ytk, moment):
    import numpy as np
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk

def fv_likelihood_statistics(samples, means, covs, weights):

    from scipy.stats import multivariate_normal
    import numpy as np

    gaussians, s0, s1,s2 = {}, {}, {}, {}
    samples = zip(range(0, len(samples)), samples)

    #print samples

    g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]
    for index, x in samples:
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in samples:
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + fv_likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + fv_likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + fv_likelihood_moment(x, probabilities[k], 2)

    return s0, s1, s2

def fv_fisher_vector_weights(s0, s1, s2, means, covs, w, T):
    import numpy as np
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fv_fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    import numpy as np
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fv_fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    import numpy as np
    return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def fv_normalize(fisher_vector):

    import numpy as np

    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))

    #return fisher_vector

def fv_fisher_vector(samples, means, covs, w):

    #from vlfeat.gmm import *
    import vlfeat.gmm as vg
    #from vlfeat.fisher import * 
    import vlfeat.fisher as vf
    from random import random
    import numpy as np

    #numData = len(samples)
    #dimension  = len(sanples[0])
    (numData, dimension) = samples.shape
    numClusters = len(w)
    flags = vf.VL_FISHER_FLAG_FAST
    
    #array_data = np.asarray(samples, dtype=np.float32)  # or array_data = samples
    
    enc = (vg.c_float *(2*dimension*numClusters))()
    nt = vf.vl_fisher_encode(enc, vg.VL_TYPE_FLOAT, means, dimension, numClusters, covs, w, samples, numData, flags)
    fv = [enc[i] for i in range(2*dimension*numClusters)]

    #print 'fv = ', fv

    return np.asarray(fv, dtype=np.float32)

def le_descritores_old(sift_folder, sift_type, subset, tipo=1):

    import os
    import numpy as np

    #n_folds = len(folds)
    #Alterei para que inclua nas imagens da galeria i no conj. train, de forma a que as
    # imagens correspondentes ao probe existam na galeria (train)
    #    for i in range(n_folds):
    #        train = folds[i][0]
    #        for j in range(n_folds):
    #            if j!=i :
    #                train = train + folds[j][0]+folds[j][1]+folds[j][2]
    #
    #        n_train = len(train)

    ch = 0
    ds = []
    id_ds = []

    for image in subset:

        #fname = os.path.join(sift_folder, image[:-3] + sift_type + '_ds')
        #ds1 = (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.uint8) #,skiprows=1)

        aux, _, _, _ = ds_read(sift_folder, image, sift_type)
        ds1 = np.asarray(aux, dtype=np.float32)    

        if tipo == 1:
            if ch == 0:
                ch = 1
                ds = []
                ds.append(ds1)
                id_ds.append(ds1.shape[0])
            else:
                ds.append(ds1)
                id_ds.append(ds1.shape[0])
        else:
            if ch == 0:
                ch = 1
                ds = np.empty_like(ds1)
                ds[:] = ds1
                id_ds.append(ds1.shape[0])
            else:
                print (ds.shape, ds1.shape)
                ds = np.concatenate((ds, ds1), axis=0)
                id_ds.append(ds1.shape[0])

    return ds, id_ds

#%%

def mask_read(mask_folder, image_name):

    import cv2 as cv
    import os.path
    from os import path

    filename = image_name[:-4]+'_mask_'
    cm = 1
    
    while True:
        
        ind = ('00'+str(cm))[-2:]
        filename_full = mask_folder + filename + ind +'.jpg'
        #print(filename_full)
        
        if path.exists(filename_full):
            
            aux = cv.imread(filename_full, cv.IMREAD_GRAYSCALE)
            
            # aplicada a mesma transformação utilizada qdo da geração dos descritores
            (w,h) = aux.shape
            if(w*h)<10000:
                res = cv.resize(aux,None,fx=1.5, fy=1.5, interpolation = cv.INTER_CUBIC)
                aux = res
            
            if cm==1:
                _ , mask = cv.threshold(aux,127,1,cv.THRESH_BINARY)
                
            else:
                #print(cm)
                mask = cv.add(mask, cv.threshold(aux,127,1,cv.THRESH_BINARY)[1])
            
            cm = cm + 1
        else:
            #print('Terminou de montar a máscara!')
            break
            
    return mask

#%%
# Le descritores conforme a mascara    
def le_descritores(sift_folder, sift_type, subset,  tipo=1):

    import os
    import numpy as np

    mask_folder = '/Projeto/dataset/masks/'
    
    #n_folds = len(folds)
    #Alterei para que inclua nas imagens da galeria i no conj. train, de forma a que as
    # imagens correspondentes ao probe existam na galeria (train)
    #    for i in range(n_folds):
    #        train = folds[i][0]
    #        for j in range(n_folds):
    #            if j!=i :
    #                train = train + folds[j][0]+folds[j][1]+folds[j][2]
    #
    #        n_train = len(train)

    ch = 0
    ds = []
    id_ds = []

    for image in subset:

        #fname = os.path.join(sift_folder, image[:-3] + sift_type + '_ds')
        #ds1 = (np.loadtxt(open(fname,"r"),delimiter=",")).astype(np.uint8) #,skiprows=1)

        mask = mask_read(mask_folder, image)
        
        kp1, kp_type, kp_num = kp_read(sift_folder,image, sift_type)
        aux, _, _, _ = ds_read(sift_folder, image, sift_type)

        ds1_aux = []
        for kpt, des in zip(kp1,aux):
            if mask[int(kpt.pt[1]), int(kpt.pt[0])] == 1:
                ds1_aux.append(des)

        ds1 = np.asarray(ds1_aux, dtype=np.float32)
        
        if tipo == 1:
            if ch == 0:
                ch = 1
                ds = []
                ds.append(ds1)
                id_ds.append(ds1.shape[0])
            else:
                ds.append(ds1)
                id_ds.append(ds1.shape[0])
        else:
            if ch == 0:
                ch = 1
                ds = np.empty_like(ds1)
                ds[:] = ds1
                id_ds.append(ds1.shape[0])
            else:
                print (ds.shape, ds1.shape)
                ds = np.concatenate((ds, ds1), axis=0)
                id_ds.append(ds1.shape[0])

    return ds, id_ds

#%%
def bov_codebook_gera(l_sift, nc, tipo):

    if tipo == 1:

        # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit
        from sklearn.cluster import KMeans

        est = KMeans(n_clusters=nc, init='k-means++', n_init=10, max_iter=100,
                     tol=0.0001, precompute_distances='auto', verbose=0,
                     random_state=None, copy_x=True, n_jobs=4)
        est.fit(l_sift)
        labels = est.labels_
        centers = est.cluster_centers_

    elif tipo == 2:

        from sklearn.cluster import MiniBatchKMeans

        est = MiniBatchKMeans(n_clusters=nc, init='k-means++', max_iter=100,
                              batch_size=3*nc, verbose=0, compute_labels=True,
                              random_state=None, tol=0.0, max_no_improvement=10,
                              init_size=None, n_init=3, reassignment_ratio=0.01)
        print (nc, len(l_sift))
        est.fit(l_sift)
        labels = est.labels_
        centers = est.cluster_centers_

    else:

        import random
        from scipy.cluster.vq import vq
        import numpy as np

        list_of_random_items = random.sample(np.arange(l_sift.shape[0]), nc)
        l_centroids = []
        for i in list_of_random_items:
            l_centroids.append(l_sift[i])

        centers = np.asarray(l_centroids)
        labels, _ = vq(l_sift, centers)

    return (centers, labels)

#%%
def bov_histogramas_gera(labels, dist_cen, id_ds, k, nomes_imagens, vis=False, th=400):

    from matplotlib import pyplot as plt
    import numpy as np

    #fv = np.vectorize(f)

    hists = []
    i = 0


    for j in range(len(nomes_imagens)):
        
             
        #ld = X[indices[j]].tolist()
        n = id_ds[j]
        sl = labels[i:i+n]  #sl_aux
        #mask = (dist_cen[i:i+n] < th) #valor empirico que pode ser parâmetro
        #sl = sl_aux[mask]

        print(i,n,len(labels))
       
        hist, bins = np.histogram(sl, bins=k, range=(0, k), normed=False,
                                  weights=None, density=True)

        if j == (len(nomes_imagens)-1):
            print (nomes_imagens[j], id_ds[j])
            
        if vis == True:
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            plt.title("Histogram "+nomes_imagens[j])
            plt.xlabel("Visual Word")
            plt.ylabel("Frequency")
            plt.bar(center, hist, align='center', width=width)
            plt.show()
            #print j

        hists.append(hist)
        #print hist
        i = i + n
        #j = j +1

    return hists

def bov_descritores_codifica(X, centers):
    from scipy.cluster.vq import vq

    labels,_ = vq(X,centers)

    return labels

 #%%
def gera_svm_dat(i, classes, hists, dt, test):
    """Gera o arquivo no formato de entrada da LIBSVM.
    Parâmetros
    ----------
    classes : lista
        Classe correspondente a cada imagem.
    x_squared_norms : lista
        histograma de cada imagem.
    """
    if test == True:
        arquivo = 'test'
    else:
        arquivo = 'training'

    path = "/Projeto/Projetos/dat_artigo/"

    rs_file = open(path + 'svm_' + str(i) + '_' + arquivo + '_' +dt+'.txt', 'w')

    dim = len(hists[0])
    t = len(hists)
    for (classe, hist) in zip(classes, hists):
        #print c, h
        rs_file.write(str(classe))
        for i in range(dim):
            rs_file.write(' ' + str(i + 1) + ':' + str(hist[i]))
        if t > 1:
            rs_file.write('\n')
        t = t - 1
    rs_file.close()

#%%%
def gera_opf_dat(i, classes, hists, dt, test, tipo=1):
    """Gera o arquivo no formato de entrada da LIBOPF.
    Parâmetros
    ----------
    classes : lista
        Classe correspondente a cada imagem.
    hists : lista
        histograma de cada imagem.
    tipo : inteiro
        tipo do arquivo de saída (texto=1, binário=2)
    """
    from struct import pack

    nc = 1
    ant = classes[0]
    for i in classes:
        print (i, nc)
        if i!=ant:
            nc = nc+1
            ant = i
    print (classes)
    print ("classes = ",nc)

    path = "/Projeto/Projetos/dat_artigo/"

    if test == True:
        arquivo = 'test'
    else:
        arquivo = 'training'
    if tipo == 1:
        rs_file = open(path + 'opf_' + str(i) + '_'  + arquivo + '_' + dt + '.txt', 'w')

        dim = len(hists[0])
        n_hists = len(hists)
        linha = str(n_hists) + ' ' + str(nc) + ' ' + str(dim) + '\n'
        rs_file.write(linha)
        k = 0
        nc = 1
        ant = classes[0]
        for (classe, hist) in zip(classes, hists):
            if classe!=ant:
               nc = nc+1
               ant = classe
            #print c, h
            linha = str(k) + ' ' + str(nc)

            print (str(k)+' '+str(nc)+' '+str(len(hists[k]))+'\n')

            for i in range(dim):
                linha = linha + ' ' + str(hist[i])

            rs_file.write(linha)
            if n_hists > 1:
                rs_file.write('\n')
            n_hists = n_hists - 1
            k = k + 1
    else:
        rs_file = open(path + 'opf_' + arquivo + '_' +dt+'.dat', 'wb')

        dim = len(hists[0])
        n_hists = len(hists)
        linha = pack('iii', n_hists, 45, dim)
        k = 0
        for (classe, hist) in zip(classes, hists):
            #print c, h
            linha = pack('ii', k, classe)
            for i in range(dim):
                linha = linha + pack('f', hist[i])
            rs_file.write(linha)
            k = k + 1

    rs_file.close()
    
#%%
def bov_histogramas_grava(arquivo, hists, dt):

    resultFile = open(arquivo, 'w')
    i = len(hists)
    for h in hists:
        line = (''.join(str(e) + ", " for e in h.tolist()))[:-2]
        resultFile.write(line)
        if i > 0:
            resultFile.write("\n")
        i = i - 1

    resultFile.close()

#%%
def bov_codebook_grava(arquivo, centers, dt):

    resultFile = open(arquivo, 'w')
    i = centers.shape[0]
    for c in centers:
        line = (''.join(str(e) + ", " for e in c.tolist()))[:-2]
        resultFile.write(line)
        if i > 0:
            resultFile.write("\n")
        i = i - 1

    resultFile.close()
