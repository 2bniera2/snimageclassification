import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA

facebook = np.uint8(np.load('transformed/facebook_(224, 224).npy'))
instagram = np.uint8(np.load('transformed/instagram_(224, 224).npy'))
orig = np.uint8(np.load('transformed/orig_(224, 224).npy'))
telegram = np.uint8(np.load('transformed/telegram_(224, 224).npy'))
twitter = np.uint8(np.load('transformed/twitter_(224, 224).npy'))
whatsapp = np.uint8(np.load('transformed/whatsapp_(224, 224).npy'))


X = np.concatenate((
    facebook,
    instagram,
    orig,
    telegram,
    twitter,
    whatsapp
))

n, r, c = X.shape

X = X.reshape((n, r*c))

y = np.array([0,1,2,3,4,5])
y = np.repeat(y, 143)

pca = PCA(2)
kernel_pca = KernelPCA(
    n_components=2,kernel="poly", gamma=50000
)

Xt = kernel_pca.fit_transform(X)

plot = plt.scatter(Xt[:,0], Xt[:,1], c=y)

plt.show()
# cmap = 'gnuplot2'

# plt.subplot(2, 6, 1)
# plt.imshow(facebook[0], cmap=cmap, vmin=0, vmax=255)
# plt.subplot(2, 6, 2)
# plt.imshow(instagram[0], cmap=cmap, vmin=0, vmax=255)
# plt.subplot(2, 6, 3)
# plt.imshow(orig[0], cmap=cmap, vmin=0, vmax=255)
# plt.subplot(2, 6, 4)
# plt.imshow(telegram[0], cmap=cmap, vmin=0, vmax=255)
# plt.subplot(2, 6, 5)
# plt.imshow(twitter[0], cmap=cmap, vmin=0, vmax=255)
# plt.subplot(2, 6, 6)
# plt.imshow(whatsapp[0], cmap=cmap, vmin=0, vmax=255)

# plt.subplot(2, 6, 7)
# plt.imshow(facebook[1], cmap=cmap, vmin=0, vmax=255)
# plt.subplot(2, 6, 8)
# plt.imshow(instagram[1], cmap=cmap, vmin=0, vmax=255)
# plt.subplot(2, 6, 9)
# plt.imshow(orig[1], cmap=cmap, vmin=0, vmax=255)
# plt.subplot(2, 6, 10)
# plt.imshow(telegram[1], cmap=cmap, vmin=0, vmax=255)
# plt.subplot(2, 6, 11)
# plt.imshow(twitter[1], cmap=cmap, vmin=0, vmax=255)
# plt.subplot(2, 6, 12)
# plt.imshow(whatsapp[1], cmap=cmap, vmin=0, vmax=255)


# plt.colorbar


# plt.show()
