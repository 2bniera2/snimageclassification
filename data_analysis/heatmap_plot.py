import matplotlib.pyplot as plt
import numpy as np

facebook_his = np.load('histogram_processed/facebook_0.npy')
instagram_his = np.load('histogram_processed/instagram_0.npy')
orig_his = np.load('histogram_processed/orig_0.npy')
telegram_his = np.load('histogram_processed/telegram_0.npy')
twitter_his = np.load('histogram_processed/twitter_0.npy')
whatsapp_his = np.load('histogram_processed/whatsapp_0.npy')

facebook_his_64 = np.load('histogram_processed/facebook_64.npy')
instagram_his_64 = np.load('histogram_processed/instagram_64.npy')
orig_his_64 = np.load('histogram_processed/orig_64.npy')
telegram_his_64 = np.load('histogram_processed/telegram_64.npy')
twitter_his_64 = np.load('histogram_processed/twitter_64.npy')
whatsapp_his_64 = np.load('histogram_processed/whatsapp_64.npy')


cmap = 'gnuplot2'

plt.subplot(6, 2, 1)
plt.imshow(facebook_his, cmap=cmap)
plt.title('facebook full image size')
plt.subplot(6, 2, 2)
plt.imshow(facebook_his_64, cmap=cmap)
plt.title('facebook patch size 64x64')

plt.subplot(6, 2, 3)
plt.imshow(instagram_his, cmap=cmap)
plt.title('instagram full image size')
plt.subplot(6, 2, 4)
plt.imshow(instagram_his_64, cmap=cmap)
plt.title('instagram patch size 64x64')

plt.subplot(6, 2, 5)
plt.imshow(orig_his, cmap=cmap)
plt.title('original full image size')
plt.subplot(6, 2, 6)
plt.imshow(orig_his_64, cmap=cmap)
plt.title('original patch size 64x64')

plt.subplot(6, 2, 7)
plt.imshow(telegram_his, cmap=cmap)
plt.title('telegram full image size')
plt.subplot(6, 2, 8)
plt.imshow(telegram_his_64, cmap=cmap)
plt.title('telegram patch size 64x64')

plt.subplot(6, 2, 9)
plt.imshow(twitter_his, cmap=cmap)
plt.title('twitter full image size')
plt.subplot(6, 2, 10)
plt.imshow(twitter_his_64, cmap=cmap)
plt.title('twitter patch size 64x64')

plt.subplot(6, 2, 11)
plt.imshow(whatsapp_his, cmap=cmap)
plt.title('whatsapp full image size')
plt.subplot(6, 2, 12)
plt.imshow(whatsapp_his_64, cmap=cmap)
plt.title('whatsapp patch size 64x64')

plt.show()
