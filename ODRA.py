
import numpy as np
import pandas as pd

from scipy.spatial import distance
import math

class ODRA:
  def __init__(self,df):
    self.data_df = df
    self.data = np.array(self.data_df.iloc[:,:-1])
    self.y = np.array(self.data_df.iloc[:,-1])
    self.k = len(self.data_df)
    self.sparseMatrix = []
  
  def distance_oneD(self,i,j):
    return math.fabs(i-j)
  
  def nn_1d(self,i,j,oneD_array):
    nearest_neighbors = []
    for cnt in range(len(oneD_array)):
      nearest_neighbors.append((self.distance_oneD(oneD_array[i],oneD_array[cnt]),cnt))
    return nearest_neighbors
  
  def k_plus1_neighbors(self,i,j,oneD_array):
    nearest_neighbors = self.nn_1d(i,j,oneD_array)
    nearest_neighbors = sorted(nearest_neighbors, key=lambda x: x[0])
    return nearest_neighbors[:self.k+1]
  
  def center_nn(self,nn,oneD_array):
    sum = 0
    for i in nn:
      sum += oneD_array[i[1]]
    return float(sum/(self.k+1))
  
  def sparseness_degree(self,i,j):
    oneD_array = self.data[:,j]
    #print(oneD_array)
    kplus1_nn = self.k_plus1_neighbors(i,j,oneD_array)
    c_ij = self.center_nn(kplus1_nn,oneD_array)
    sum = 0
    for cnt in kplus1_nn:
      #print(oneD_array[cnt[1]])
      sum += (oneD_array[cnt[1]] - c_ij)**2
    return float(sum/(self.k+1))
    #return c_ij
  
  def compute_sparseness(self,i):
    sprse = []
    for m in range(self.data.shape[1]):
      sprse.append(self.sparseness_degree(i,m))
    return sprse 

  def construct_sparse_matrix(self):
    for n in range(self.data.shape[0]):
      self.sparseMatrix.append(self.compute_sparseness(n))
