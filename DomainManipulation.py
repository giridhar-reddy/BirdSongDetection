import numpy as np
from scipy import fft, arange
import librosa
import mdp

def frequency_spectrum(x, sf, samples=None):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    k = arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    frqarr, xabs = functionRollingMean(frqarr, abs(x), 0, samples=samples)

    return frqarr,xabs


def functionRollingMean(xArr, yArr, window, samples=None):
    window = int(window)

    xValArr = rollingMean(xArr, window, samples=samples)
    yValArr = rollingMean(yArr, window, samples=samples)

    return xValArr, yValArr


def rollingMean(x, window, samples=None):
    window = int(window)
    xValArr = []

    if samples!=None:
        s = samples
        window = len(x)/samples
    else:
        s = int(len(x) / window)

    for i in range(s):
        val = np.mean(x[int(i * window): int(np.ceil((i + 1) * window))])
        xValArr.append(val)
    xValArr = np.array(xValArr)

    return xValArr


def mfcc(y, n_mfcc, sr=22050):
    mfcc_feat = librosa.feature.mfcc(n_mfcc=n_mfcc, y=y, sr=sr)
    return mfcc_feat

def sfc(y, n_sfc):
    sfa = mdp.nodes.SFANode(output_dim=n_sfc)
    sfa.train(y)

    return sfa(y)

def sfc_component(y, n_sfc):
    sfa = mdp.nodes.SFANode(output_dim=n_sfc)
    sfa.train(y)

    return sfa


if 5==None:
    print("hello")