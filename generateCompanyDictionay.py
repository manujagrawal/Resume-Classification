import numpy as np
import csv

f = open('company_labels.csv', 'r')
reader = csv.reader(f)

companies = []
labels = []

for row in reader:
	companies.append(row[0])
	row2=[float(i) for i in row[1:] ]
	labels.append(row2)

companies=np.array(companies)
labels=np.array(labels)

print companies.shape  
print labels.shape 

