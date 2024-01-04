import numpy as np
from matplotlib import pyplot as plt
import PIL 
from PIL import Image

import os
image_dir = "./images"
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
image_files.remove(".DS_Store")

num_images = len(image_files)
num_rows = int(np.sqrt(num_images))
num_cols = int(np.ceil(num_images / num_rows))
fig, axes = plt.subplots(num_rows, num_cols)

for i, ax in enumerate(axes.flat):
    if i < num_images:
        image_path = image_dir + "/" + image_files[i]
        image = np.array(Image.open(image_path))
        ax.imshow(image)
        ax.axis('off')
fig.subplots_adjust(hspace=0.1, wspace=0.1)
# plt.show()


################################################
compress_rates = [0.1, 0.5, 1.5, 10]
num_cols = len(compress_rates) + 1
fig, axes = plt.subplots(1, num_cols)
image = np.array(Image.open("./images/original.jpg").convert('L'))
axes[0].imshow(image, cmap='gray')
axes[0].axis('off')

for i, compress_rate in enumerate(compress_rates):
    compress_rate /= 100
    U, S, V = np.linalg.svd(image)
    k = int(compress_rate * len(S))
    compressed_image = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
    axes[i+1].imshow(compressed_image, cmap='gray')
    axes[i+1].set_title(f'{compress_rate*100}')
    axes[i+1].axis('off')

# plt.show()


#############################################
num_cols = len(compress_rates) + 1
fig, axes = plt.subplots(1, num_cols)
image = np.array(Image.open("./images/original.jpg").convert('L'))

axes[0].imshow(image, cmap='gray')
axes[0].axis('off')

for i, compress_rate in enumerate(compress_rates):
    compress_rate /= 100
    F = np.fft.fft2(image)
    Fabs = np.abs(F)
    Fabs_sorted = np.sort(Fabs.flatten())[::-1]
    k = int(compress_rate * len(Fabs_sorted))
    threshold = Fabs_sorted[k]
    F_filtered = F * (Fabs >= threshold)
    compressed_image = np.real(np.fft.ifft2(F_filtered))
    axes[i+1].imshow(compressed_image, cmap='gray')
    axes[i+1].set_title(f'{compress_rate*100}')
    axes[i+1].axis('off')

# plt.show()
ranks = [5, 20, 50]
num_cols = len(ranks) + 1
fig, axes = plt.subplots(1, num_cols)
image_noisy = np.array(Image.open('./images/noisy.jpg'))

axes[0].imshow(image, cmap='gray')
axes[0].axis('off')

for i, k in enumerate(ranks):
    U, S, V = np.linalg.svd(image_noisy,full_matrices=False)
    denoised_image = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
    axes[i+1].imshow(denoised_image, cmap='gray')
    axes[i+1].set_title(f'{k}')
    axes[i+1].axis('off')
# plt.show()

#######################################
radii = [500, 1000, 3000, 5000]
num_cols = len(radii) + 1
fig, axes = plt.subplots(1, num_cols)
image_noisy = np.array(Image.open("./images/noisy.jpg"))

axes[0].imshow(image, cmap='gray')
axes[0].axis('off')

for i, radius in enumerate(radii):
    fft_image = np.fft.fft2(image_noisy)
    center_x, center_y = np.array(fft_image.shape) // 2
    mask = np.zeros(fft_image.shape)
    mask[(center_x-radius):(center_x+radius), (center_y-radius):(center_y+radius)] = 1
    fft_image_filtered = fft_image * mask
    denoised_image = np.fft.ifft2(fft_image_filtered).real

    axes[i+1].imshow(denoised_image, cmap='gray')
    axes[i+1].set_title(f'{radius}')
    axes[i+1].axis('off')

plt.show()