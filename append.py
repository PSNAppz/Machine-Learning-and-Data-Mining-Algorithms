import csv
filename = 'Dataset/ndrug/Morgan.csv'
output = 'Dataset/ndrug/Morgan1.csv'

with open(filename,'r') as csvinput:
    with open(output, 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        for row in csv.reader(csvinput):
            writer.writerow(row+['0'])