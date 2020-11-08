import csv
filename = 'Dataset/drug/Toptor.csv'
output = 'Dataset/drug/Toptor1.csv'

with open(filename,'r') as csvinput:
    with open(output, 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        for row in csv.reader(csvinput):
            writer.writerow(row+['1'])