from __future__ import division
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
import os
import sys
from dotenv import load_dotenv

nArray = 0
mtx = 0

# environment variable
load_dotenv()
DATASET_FOLDER = os.getenv('DATASET')
DATASET_OUTPUT_FOLDER = os.getenv('DATASET_OUTPUT')

def call_function():
    try:
        float_formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})
        # np.set_printoptions(threshold='nan')
        k=int(input("Enter the value for k: "))
        build_rotationtree_model(k)
    except:
        e = sys.exc_info()[0]
        print( "<p>Error: %s</p>" % e )


def PCA(newArray,mtx):
  X = np.array(newArray)
  columns = mtx.shape[1]
  columns -= 1
  rows = mtx.shape[0]
  nCol = X.shape[1]
  Y = np.array(mtx)[:,columns]
  Mean= [0 for a in range(nCol)]
  countX1 = 0
  countX2 = 0
  x1=[[0 for w in range(X.shape[1])] for z in range(rows)]
  for i in range(rows):
    for j in range(nCol):
      x1[i][j] = X[i][j]
  for i in range(nCol):
    for j in range(rows):
      Mean[i] +=x1[j][i]
    Mean[i] /= rows  
  for i in range(nCol):
    for j in range(rows):
      x1[j][i] -= Mean[i]  
  cov = np.cov(np.array(x1).T)
  eigval, eigvec = LA.eig(cov)
  x1 = np.array(x1).T
  pca = np.dot(eigvec,x1)
  return pca

 
def build_rotationtree_model(k):
  mtx = genfromtxt("/".join([DATASET_FOLDER, 'dnd/MACCS166.csv']), delimiter=',')
  #Length of attributes (width of matrix)
  a = mtx.shape[1] 
  a -= 1
  #Seperation limit
  limit = int(a/k)
  newArray =[[0 for x in range(limit)] for y in range(len(mtx))]
  #Height of matrix(total rows)
  b = mtx.shape[0]
  #Sparse matrix
  sparseMat = [[0 for x in range(a)] for y in range(a)]
  #Starting of sub matrix
  start = 0
  #Ending of sub matrix
  end = int(a/k)
  cond = end
  m = 0
  n = 0
  pos = 0
  counter = 0
  ext = 0
  add = 0
  #Loop
  while(counter < k):
      if(counter == k-1):
        add = a - (k*cond)
        end = end + add
        newArray =[[0 for x in range(limit+add)] for y in range(len(mtx))]
        ext = 1
      counter += 1
      for i in range(0,b):
          for j in range(start,end):
              newArray[i][n] = mtx[i][j]
              n = n+1    
          n=0
      invPooled = np.array(PCA(newArray,mtx)) 
      sparse(pos,invPooled,limit,sparseMat,ext,add)
      pos = pos + limit
      start = end
      end = end + limit 
  originMTX = np.delete(mtx,a,axis=1)
  sparseMat = np.matrix(sparseMat)
  result = np.array(originMTX * sparseMat)
  Ycol = np.array(mtx)[:,a]
  result = result.round(decimals = 2)
  final = np.concatenate((result,Ycol.reshape(Ycol.shape[0],1).astype(int)),axis=1)
  np.savetxt("/".join([DATASET_OUTPUT_FOLDER, 'PCAdata.csv']),final,fmt='%10.2f',delimiter=",")


def sparse(pos,invPooled,limit,sparseMat,ext,add):
  e = pos
  f = pos
  if(ext):
    limit +=add
  for s in range(limit):
    for t in range(limit):
      sparseMat[e][f] = invPooled[s][t]
      f = f+ 1
    e = e + 1
    f = pos
  return sparseMat  