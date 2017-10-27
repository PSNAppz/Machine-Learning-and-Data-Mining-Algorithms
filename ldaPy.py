from __future__ import division
import numpy as np
def main():
	build_rotationtree_model(2)

def lda(newArray):

 Curvature = np.array([2.95,2.53,3.57,3.16,2.58,2.16,3.27])  
 Diameter = np.array([6.63,7.79,5.65,5.47,4.46,6.22,3.52])            
 QCR = np.array([1,1,1,1,0,0,0])
 Mean_X1_A=0
 Mean_X1_B=0
 Mean_X2_A=0
 Mean_X2_B=0
 countX1=0
 countX2=0
 x=[[0 for x in range(2)] for y in range(len(newArray))]
 x1=[[0 for w in range(2)] for z in range(len(newArray))]
 x2=[[0 for e in range(2)] for q in range(len(newArray))]
 y2=[[0 for d in range(2)] for c in range(len(newArray))]
 y2=[[0 for u in range(2)] for y in range(len(newArray))]
 A1=[[0 for v in range(2)] for k in range(4)]
 A2=[[0 for f in range(2)] for j in range(3)]

 for i in range(len(Curvature)):
   x[i][0]= Curvature[i]
   x[i][1]= Diameter[i]

 for i in range(len(QCR)):
 	if(QCR[i]==1):
 		x1[i][0]=Curvature[i]
 		x1[i][1]=Diameter[i]
 	else:
 		x2[i][0]=Curvature[i]
 		x2[i][1]=Diameter[i]

#Mean of each attribute
 for i in range(len(QCR)):
	if(QCR[i]==1):
		Mean_X1_A+=x1[i][0]
		Mean_X1_B+=x1[i][1]
		countX1+=1
	else:
		Mean_X2_A+=x2[i][0]
		Mean_X2_B+=x2[i][1]
		countX2+=1	

 Mean_X1_A/=countX1
 Mean_X1_B/=countX1
 Mean_X2_A/=countX2
 Mean_X2_B/=countX2		

 GlobalMean_A=(Mean_X1_A + Mean_X2_A)/2
 GlobalMean_B=(Mean_X1_B + Mean_X2_B)/2

 #Subtraction of global mean from each of data dimensions.
 for i in range(len(QCR)):
 	if(QCR[i]==1):
 		x1[i][0]-=GlobalMean_A
 		x1[i][1]-=GlobalMean_B
 	else:
 		x2[i][0]-=GlobalMean_A
 		x2[i][1]-=GlobalMean_B
 a=0 
 b=0		
 for i in range(len(QCR)):
 	if(QCR[i]==1):
 		A1[b][0]= x1[i][0]
 		A1[b][1]= x1[i][1]
 		b+=1 		

 	else:
 		A2[a][0]= x2[i][0]
  		A2[a][1]= x2[i][1]
  		a+=1
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
 print("InvPooled")
 print(invPooled)

 
def build_rotationtree_model(k):
 mtx =np.array([[2.95,6,63,23],[2,53,7,79],[3.57,5,65,32],[3.16,5,47,34],[21,2.58,4,46],[3.1,2.16,6,22],[3.5,3.27,3,52],[12,2.56,4,41]])	
 #Length of attributes (width of matrix)
 a = mtx.shape[1]
 newArray =[[0 for x in range(k)] for y in range(len(mtx))]
 #Height of matrix(total rows)
 b = mtx.shape[0]
 #Seperation limit
 limit = int(a/k)
 #Starting of sub matrix
 start = 0
 #Ending of sub matrix
 end = int(a/k)
 m=0
 n=0
  #Loop
 while(end <= a):
 	for i in range(0,b):
 		for j in range(start,end):
 			newArray[m][n] = mtx[i][j]
 			n = 1
 		m=m+1
 		n=0
 	print(newArray)
 	lda(newArray)
 	#Call LDA function and add the result to Sparse Matrix
 	#sparseMat = LDA(newArray) Should be inside a loop
 	m = 0
 	start = end
 	end = end + limit


if __name__ == '__main__':
	main()
