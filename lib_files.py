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
def kp_write(file_path, filename, tipo, kp):
    import csv

    #file_path = '/media/sf_Projeto/Projetos/dataset/descriptors'

    if not (kp is None) :
        with open(file_path + '/' + filename, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([tipo, len(kp)])
            for point in kp:
                spamwriter.writerow([point.pt[0], point.pt[1], point.size, point.angle, point.response, point.octave, point.class_id])


#%%
def des_write(file_path, filename, tipo, des):
    import csv

    #file_path = '/media/sf_Projeto/Projetos/dataset/descriptors'

    if not(des is None) :
        with open(file_path + '/' + filename, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([tipo, len(des), len(des[0])])
            for desc in des:
                spamwriter.writerow(desc)
