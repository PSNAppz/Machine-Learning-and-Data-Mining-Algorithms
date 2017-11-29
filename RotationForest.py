from __future__ import division
import numpy as np
from numpy import genfromtxt

def main():
  build_rotationtree_model(2)



def LDA(newArray,mtx):
  X = np.array(newArray)
  columns = mtx.shape[1]
  columns -= 1
  rows = mtx.shape[0]
  nCol = X.shape[1]
  Y = np.array(mtx)[:,columns]
  Mean_X1= [0 for a in range(nCol)]
  Mean_X2= [0 for b in range(nCol)]
  countX1 = 0
  countX2 = 0
  x1=[[0 for w in range(X.shape[1])] for z in range(rows)]
  x2=[[0 for e in range(X.shape[1])] for q in range(rows)]
  GlobalMean = [0 for a in range(nCol)]
  for i in range(rows):
    if(mtx[i][columns]==1):
      for j in range(nCol):
        x1[i][j] = X[i][j]
    else:
       for k in range(nCol):
         x2[i][k] = X[i][k]
  #MEAN
  for i in range(nCol):
    for j in range(rows):
      if(mtx[j][columns]==1):
        Mean_X1[i] +=x1[j][i]
        countX1+=1
      else:
        Mean_X2[i]+=x2[j][i]
        countX2+=1  
  #Global Mean
  for j in range(nCol):
    Mean_X1[j]/= countX1
    Mean_X2[j]/= countX2   
  for k in range(nCol):
    GlobalMean[k] = (Mean_X1[k] + Mean_X2[k])/2
  #print("MEAN",GlobalMean) 
  for i in range(nCol):
    for j in range(rows):
      if(mtx[j][columns] == 1):
        x1[j][i] -= GlobalMean[i]   
      else:
        x2[j][i] -= GlobalMean[i]
  A1=[[0 for v in range(nCol)] for k in range(countX1)]
  A2=[[0 for f in range(nCol)] for j in range(countX2)]     
  for i in range(rows):
    if(mtx[i][columns]==1):
      for j in range(nCol):
        A1[i][j] = X[i][j]
    else:
       for k in range(nCol):
         A2[i][k] = X[i][k]
               
  TransMat_A = np.asarray(A1)
  TransMat_B = np.asarray(A2)
  CovMat_A = np.cov(np.matrix.transpose(TransMat_A))
  CovMat_B = np.cov(np.matrix.transpose(TransMat_B))     
  pooled =[[0 for f in range(len(CovMat_A))] for j in range(len(CovMat_A))]
  for i in range(len(CovMat_A)):
    for j in range(len(CovMat_A)):
      pooled[i][j]=(countX1/(countX1+countX2))*CovMat_A[i][j]+(countX2/(countX1+countX2))*CovMat_B[i][j]
  invPooled = [[0 for r in range(len(CovMat_A))] for i in range(len(CovMat_A))]
  invPooled = np.matrix(pooled).I
  return invPooled

 

 
def build_rotationtree_model(k):
  mtx = genfromtxt('heart.csv', delimiter=',')
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
  #Loop
  while(counter < k):
      counter += 1
      for i in range(0,b):
          for j in range(start,end):
              newArray[m][n] = mtx[i][j]
              n = n+1
          m=m+1
          n=0
      invPooled = np.array(LDA(newArray,mtx)) 
      sparse(pos,invPooled,limit,sparseMat)
      pos = pos + limit
      m = 0
      start = end
      end = end + limit 
  originMTX = np.delete(mtx,a,axis=1)
  sparseMat = np.matrix(sparseMat)
  print(originMTX * sparseMat)    




def sparse(pos,invPooled,limit,sparseMat):
  e = pos
  f = pos
  for s in range(limit):
    for t in range(limit):
      sparseMat[e][f] = invPooled[s][t]
      f = f+ 1
    e = e + 1
    f = pos
  return sparseMat  

 



if __name__ == '__main__':
  main()