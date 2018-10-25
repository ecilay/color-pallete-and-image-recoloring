import numpy as np
import scipy.misc
from skimage import color
from sklearn import cluster
import matplotlib.pyplot as plt
import argparse
import utils

use_lab = False

def quantize(raster, n_colors):
    width, height, depth = raster.shape
    reshaped_raster = np.reshape(raster, (width * height, depth))

    model = cluster.KMeans(n_clusters=n_colors)
    labels = model.fit_predict(reshaped_raster)
    palette = model.cluster_centers_

    quantized_raster = np.reshape(
        palette[labels], (width, height, palette.shape[1]))


    return model, quantized_raster


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--k", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())

raster = scipy.misc.imread(args["image"])
if use_lab:
	raster = color.rgb2lab(raster)
model, quantized_raster = quantize(raster, args["k"])

if use_lab:
	raster = (color.lab2rgb(raster) * 255).astype('uint8')
	quantized_raster = (color.lab2rgb(quantized_raster) * 255).astype('uint8')

hist = utils.centroid_histogram(model)
bar = utils.plot_colors(hist, model.cluster_centers_)
# if use_lab:
# 	bar = (color.lab2rgb(bar) * 255).astype('uint8')


plt.figure()
plt.axis("off")
plt.imshow(raster / 255.0)


plt.figure()
plt.axis("off")
plt.imshow(bar / 255.0)


plt.figure()
plt.axis("off")
plt.imshow(quantized_raster / 255.0)
plt.show()