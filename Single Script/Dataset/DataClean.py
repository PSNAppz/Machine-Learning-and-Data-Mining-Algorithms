#DataSet Cleaner

import csv
import numpy as np

def clean(filename):

	lines = csv.reader(open(filename, "rt"))
	dataset = list(lines)
	for i in range(len(dataset)):
		for x in range(len(dataset[i])):
			if(dataset[i][x] == ""):
				dataset[i][x] = 0
			else:
				dataset[i][x] = dataset[i][x]
	dataset = np.array(dataset)			
	np.savetxt("BF.csv",dataset,fmt='%s',delimiter=",")
			

clean("BlackFriday.csv")