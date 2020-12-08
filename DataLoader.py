import os
import librosa
import numpy as np
import shutil

def readFromFile(file, sr = 22050):
    y = np.array([])

    try:
        signal , sr = librosa.load(file, sr=sr)
    except:
        return y

    if len(signal.shape)==1:
        y = signal
    else:
        y = np.average(signal, axis=1)  # use the first channel (or take their average, alternatively)

    return y

def readFromDirectory(directory, sr=22050, filecount=None):
    files = os.listdir(directory)
    if filecount!=None:
        files = files[3:3+filecount]

    yarr = []
    for fileName in files:
        file = os.path.join(directory + "/", fileName)
        print("reading file: {}".format(file))
        new_y = readFromFile(file, sr=sr)
        yarr.append(new_y)
    return np.array(yarr)

def readFromDirectories(directory, sr=22050, dircount=None, filecount=None):
    dirs = os.listdir(directory)
    dirs = dirs[:dircount]

    y = {}
    for dir in dirs:
        dirpath = os.path.join(directory + "/", dir)
        diry = readFromDirectory(dirpath, sr=sr, filecount=filecount)
        y[dir] = diry

    return y

def splitByLabels(y, labels, targetLoc, hoprate=512):
    prevLabel = labels[0]
    previ = 0
    for i, label in enumerate(labels):
        if prevLabel != label:
            librosa.output.write_wav(os.path.join(targetLoc, str(label), "{}.wav".format(i)),
                                     y[previ * hoprate:i * hoprate, np.newaxis], sr=22050)
            prevLabel = label
            previ = i
    librosa.output.write_wav(os.path.join(targetLoc, str(label), "{}.wav".format(i)),
                             y[previ * hoprate : i * hoprate, np.newaxis], sr=22050)

def copyAllFilesDirtoDir(src,dest):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)

def mergeAudiosTo(src,dest):
    yarr = readFromDirectory(src)
    ynp = np.concatenate(yarr,axis=0)
    librosa.output.write_wav(dest, ynp, sr=22050)

def rollMFCC(mfcc,window):
    r,c = mfcc.shape
    windows = int(r/window)
    mfcc = mfcc[:window * windows,:]
    if windows==0:
        raise ValueError
    else:
        return mfcc.reshape((windows, window, c))
