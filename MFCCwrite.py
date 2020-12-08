import DataLoader,DomainManipulation
import pickle
import os
import numpy as np
import gc

directory = 'data/train_audio/'
mfcc_directory = 'data/mfcc_50/'

def getBirdMfcc(bird, n_mfcc=50):
    birddirectory = directory + "/" + bird
    files = os.listdir(birddirectory)
    mfcc = None

    for fileName in files:
        file = os.path.join(birddirectory + "/", fileName)
        print("reading file: {}".format(file))
        new_y, sr, new_t = DataLoader.readFromFile(file)
        if len(new_y)==0:
            continue
        new_mfcc = DomainManipulation.mfcc(new_y, n_mfcc)
        try:
            mfcc = np.concatenate([mfcc, new_mfcc], axis = 1)
        except ValueError:
            if mfcc == None:
                mfcc = new_mfcc
            else:
                raise ValueError
        del(new_y)
    gc.collect()
    return mfcc

birds = os.listdir(directory)
for bird in birds[147:]:
    bird_mfcc = getBirdMfcc(bird, n_mfcc=50)
    pickle.dump(bird_mfcc, open(mfcc_directory+bird+".pickle", 'wb+'))


# test pickles
ald = pickle.load(open(mfcc_directory + "aldfly.pickle",'rb'))
print(ald.shape)