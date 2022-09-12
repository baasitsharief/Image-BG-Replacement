import cv2
import numpy as np
import argparse
import os
import copy
from scipy.spatial.distance import cdist 
from itertools import groupby

# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.rcParams["figure.figsize"] = (10, 10)

parser = argparse.ArgumentParser("Performs median cut color quantization on RGB image")
parser.add_argument("-D","--depth", type=int, help="Bit-Depth for quantization")
parser.add_argument("-I","--input", type=str, help="Path to Input Image")
parser.add_argument("-O", "--output", type=str, help="Path/Filename to store output image")

class KMeansClusterer:
  def __init__(self, num_clusters):
    self.num_clusters = num_clusters
    self.centroids = []

  def fit(self, data, max_iterations=1000, seed = 10):

    #if data has more than 2 dimensions
    if(len(data.shape)>2):  
      data = data.reshape((data.shape[0], -1))
    
    #Step 1: Initialize Random Centroids from the Dataset if centroids not set
    np.random.seed(seed)
    if(len(self.centroids)==0):
      idx = np.random.choice(len(data), self.num_clusters, replace=False)
      self.centroids = data[idx, :]   

    #Step 2: Find distances from each centroid and classify each datapoint based on the distnances    
    labels = self.predict(data)

    #Step 4: Repeat the steps 
    t = range(max_iterations)

    for iters in t:

      prev_centroids = copy.deepcopy(self.centroids)

      # print(f"{iters}, previous centroids: {prev_centroids}")
      
      self.centroids = list()

      for i in range(self.num_clusters):
        temp_cent = data[labels==i].mean(axis=0)  #Recalculating center based on the clusters
        self.centroids.append(temp_cent)

      # print(self.centroids)  
      
      self.centroids = np.vstack(self.centroids)  #Updating the centroids

      labels = self.predict(data) #Recalculating labels

      # print(f"{iters}, new centroids: {self.centroids}")

      if(np.array_equal(prev_centroids, self.centroids)):       #If no change, convergence achieved thus break out of loop
        break
    
    # print(f"final centroids: {self.centroids}")
   
    return labels

  def get_distance(self, data):
    #if data has more than 2 dimensions
    if(len(data.shape)>2):  
      data = data.reshape((data.shape[0], -1))
    return cdist(data, self.centroids ,'euclidean')

  def predict(self, data):
    if(len(data.shape)>2):  
      data = data.reshape((data.shape[0], -1))
    distances = cdist(data, self.centroids ,'euclidean')
    labels = np.array([np.argmin(i) for i in distances])
    return labels

def quantize(img, bit_depth, seed = 52):
  
  #Storing indexes and flattened image
  ind_pos = list()
  flat_arr = list()
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      pixel = np.zeros((2,))
      flat_arr.append(np.array([img[i,j,0], img[i,j,1], img[i,j,2]]))
      pixel[0] = i
      pixel[1] = j 
      ind_pos.append(pixel)
  ind_pos = np.array(ind_pos, dtype=int)
  flat_arr = np.array(flat_arr)

  #K-Means Clustering on flattened image
  k = 2**bit_depth
  clf = KMeansClusterer(k)
  labels = clf.fit(flat_arr, seed=seed)

  #Storing result in a new image
  new_img = np.zeros(img.shape)
  for i in range(ind_pos.shape[0]):
    new_img[ind_pos[i, 0], ind_pos[i, 1]] = clf.centroids[labels[i]]
  new_img = new_img.astype(np.uint8)
  
  return new_img

def main():
  args = parser.parse_args()
  bit_depth = args.depth
  input_path = args.input
  output_path = args.output
  ext = input_path.split('.')[-1]
  filename = os.path.split(input_path)[-1].split('.')[0]

  img = cv2.imread(input_path)
  q_img = quantize(img, bit_depth)
  if(output_path!=None):
    cv2.imwrite(output_path, q_img)
  else:
    cv2.imwrite(f"{filename}_{bit_depth}_bit.{ext}", q_img)

if __name__ == '__main__':
  main()