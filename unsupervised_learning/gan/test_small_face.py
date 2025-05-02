import numpy as np
import matplotlib.pyplot as plt 
array_of_pictures = np.load("small_res_faces_10000.npy")
array_of_pictures=array_of_pictures.astype("float32")/255

mean_face = array_of_pictures.mean(axis=0)

mean_face_rgb = np.zeros((mean_face.shape[0], mean_face.shape[1], 3))

mean_face_rgb[..., 0] = mean_face * 1.0
mean_face_rgb[..., 1] = mean_face * 0.5
mean_face_rgb[..., 2] = mean_face * 0.2

plt.imshow(mean_face_rgb)
plt.title("Modified Mean Face (RGB)")
plt.axis("off")
plt.show()


