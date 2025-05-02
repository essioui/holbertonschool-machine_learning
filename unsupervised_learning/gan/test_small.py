import numpy as np
import matplotlib.pyplot as plt 
array_of_pictures = np.load("small_res_faces_10000.npy")
array_of_pictures=array_of_pictures.astype("float32")/255

fig,axes=plt.subplots(10,10,figsize=(10,10))
fig.suptitle("real faces")
for i in range(100) :
    axes[i//10,i%10].imshow(array_of_pictures[i,:,:])
    axes[i//10,i%10].axis("off")
plt.show()

mean_face=array_of_pictures.mean(axis=0)
plt.imshow(mean_face)
