import numpy as np 

metadata = np.load('./98australian_big_metadata.npy')

n_samles, _ = np.shape(metadata)

count = round(n_samles/5)

index = np.arange(n_samles)
np.random.shuffle(index)
for i in range(5):
    small_data = metadata[index[i*count:(i+1)*count], :]
    np.save('./'+str(i)+'_australian_samll', small_data)