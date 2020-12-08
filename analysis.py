import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import DataLoader,DomainManipulation
import Modeling
import math
from sklearn import preprocessing

import numpy as np

yarr = DataLoader.readFromDirectory('data/train_audio/aldfly/', filecount=3)
y = yarr
t = len(y)
print("length of signal:{}".format(len(y)))
print(yarr.shape)

# amplitude
# power = DomainManipulation.rollingMean([math.fabs(yy) for yy in y], 1000)
# sns.distplot(power, hist=True, kde=False)

# mfcc
mfcc = DomainManipulation.mfcc(y,10)
mfcc = np.transpose(mfcc)
print(mfcc[30])
mfcc = preprocessing.normalize(np.abs(mfcc), norm='l2')
print(mfcc[30])
print("mfcc shape:{}".format(mfcc.shape))

# append magnitude to mfcc
# mody = [math.fabs(yy) for yy in y]
# magnitude = DomainManipulation.rollingMean(mody, window=1, samples=mfcc.shape[0])
# magnitude = np.array(magnitude).reshape((len(magnitude),1))
# mfcc = np.concatenate([mfcc,magnitude], axis=1)
# print("mfcc magnitude appended shape:{}".format(mfcc.shape))


# # clustering
# # labels = Modeling.kmeans(mfcc, 3)
# labels = Modeling.GMMmodel(mfcc, 3)
#
# # HMM
# labelseq = labels.reshape(1,len(labels))
# log_prob, hmmlabels = Modeling.HMMmodel(labelseq,3,2)
#
#
# # plot segmentation
# plt.subplot(211)
# plt.plot(t, y, 'b')
#
# t = DomainManipulation.rollingMean(t, window=1, samples=len(labels))
# labels = [0.1*label for label in labels]
# plt.subplot(212)
# plt.plot(range(len(labels)), labels, 'r')

# hmmlabels = [0.1*label for label in hmmlabels]
# plt.subplot(211)
# plt.plot(range(len(hmmlabels)), hmmlabels, 'g')

# mfcc plot
# plt.clf()
# plt.imshow(mfcc.T, interpolation='nearest', cmap=cm.coolwarm, origin='lower')

# correlation
corr = np.corrcoef(mfcc)
print(corr.shape)
plt.imshow(corr, interpolation='nearest')

# fourier
# frq, X = DomainManipulation.frequency_spectrum(y, sr, 50)
# plt.plot(frq, X, 'b')
# plt.xlabel('Freq (Hz)')
# plt.ylabel('|X(freq)|')

plt.show()