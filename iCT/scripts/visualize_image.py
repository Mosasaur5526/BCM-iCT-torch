import numpy as np
import matplotlib.pyplot as plt


vis_num = 64
npz_path = 'PATH_TO_THE_SAMPLE_NPZ_FILE'
save_path = f'{npz_path[:-4]}.png'
img = np.load(npz_path)['arr_0'][:vis_num]

samples = img.reshape((8, 8, 64, 64, 3))
samples = samples.transpose((0, 2, 1, 3, 4))
samples = samples.reshape(
    (samples.shape[0] * samples.shape[1], samples.shape[2] * samples.shape[3], samples.shape[4]))
plt.figure(figsize=(10, 10))
plt.imshow(samples)
plt.axis('off')
plt.savefig(save_path)
plt.close()

