import matplotlib.pyplot as plt
import numpy as np

facebook_his = np.load('facebook.npy')
instagram_his = np.load('instagram.npy')
orig_his = np.load('orig.npy')
telegram_his = np.load('telegram.npy')
twitter_his = np.load('twitter.npy')
whatsapp_his = np.load('whatsapp.npy')



plt.subplot(2, 3, 1)
plt.imshow(facebook_his, cmap='hot')



plt.subplot(2, 3, 2)
plt.imshow(instagram_his, cmap='hot')



plt.subplot(2, 3, 3)
plt.imshow(orig_his, cmap='hot')



plt.subplot(2, 3, 4)
plt.imshow(telegram_his, cmap='hot')



plt.subplot(2, 3, 5)
plt.imshow(twitter_his, cmap='hot')



plt.subplot(2, 3, 6)
plt.imshow(whatsapp_his, cmap='hot')

plt.show()
