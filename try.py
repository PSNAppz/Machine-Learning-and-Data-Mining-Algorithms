import csv
import random
import math
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy import genfromtxt


#from sklearn.preprocessing import LabelEncoder
#http://sebastianraschka.com/Articles/2014_python_lda.html
#https://onlinecourses.science.psu.edu/stat505/node/94
#http://people.revoledu.com/kardi/tutorial/LDA/Numerical%20Example.html
#https://jyyuan.wordpress.com/2014/02/17/linear-discriminant-analysis-and-quadratic-discriminant-analysis-for-classification/
#http://stats.stackexchange.com/questions/71489/three-versions-of-discriminant-analysis-differences-and-how-to-use-them
#http://people.revoledu.com/kardi/tutorial/Similarity/MahalanobisDistance.html
#http://www.saedsayad.com/lda.htm
datos = genfromtxt('manu.csv', delimiter=',')
# Iris-setosa --> 0
#Iris-versicolor --> 1
#Iris-virginica --> 2

def porClase(datos):
  datosClase = {}
  for i in range(len(datos)):
    vector = datos[i]
    if (vector[-1] not in datosClase):
      datosClase[vector[-1]] = []
    datosClase[vector[-1]].append(vector)
  return datosClase

def mean(feature):
  return sum(feature)/float(len(feature))

def medias(features):
  medias = []
  for i in features:
    medias.append(mean(i))
  return medias

def elemento(Entrenamiento):
  elementos = [mean(atributo) for atributo in zip(*Entrenamiento)]
  del elementos[-1]
  return elementos

def poolcov(features,Medias):
  xn =[]
  y = []
  ftemp = list(zip(*features))
  del ftemp[-1]
  for v in range(len(ftemp)):
    ftemp[v] = list(ftemp[v])

  for i in range(len(ftemp)):
    for x in range(len(ftemp[i])):
      #xn.append(ftemp[i][x]-Medias[i])
      ftemp[i][x] = ftemp[i][x]-Medias[i]
  return ftemp

def categorias(datos):
  categorias_datos={}
  for i in range(len(datos)):
    if datos[i][-1] not in categorias_datos:
      categorias_datos[datos[i][-1]] = 1
    else:
      categorias_datos[datos[i][-1]] = categorias_datos[datos[i][-1]] + 1 
  return categorias_datos

def probCat(datos):
  pC=[]
  x = categorias(datos)
  for i in x:
     pC.append(x[i]/float(len(datos)))
  return pC

def covarianza(Pclase,Medias,pC):
  C = 0
  mediasC =[]
  ci ={}
  xin = {}
  for classValue, featuresD in Pclase.iteritems():
    mediasC.append(np.array(elemento(featuresD)))
    f = zip(*featuresD)
    xin[classValue] = np.array(zip(*poolcov(featuresD,Medias)))
  for i in xin:
    ci[i]= np.dot(xin[i].T,xin[i])/len(xin[i])
  for z in ci:
    C += ci[z]*pC[z]
  return (C,mediasC)

def main():
  pC = np.array(probCat(datos))
  Pclase = porClase(datos)
  features = zip(*datos)
  del features[-1]
  Medias = medias(features)
  C,mediasC = covarianza(Pclase,Medias,pC)
  PooledCov = inv(C) 
  test = np.array([2.95,6.63])
  print("POOLED",PooledCov)
  mT = np.array(test)[np.newaxis].T
  for i in Pclase:
    print np.dot(np.dot(mediasC[i],PooledCov),mT)-0.5*np.dot(np.dot(mediasC[i],PooledCov),mediasC[i].T) + np.log(pC[i])
  
#http://people.revoledu.com/kardi/tutorial/LDA/Numerical%20Example.html
main()
