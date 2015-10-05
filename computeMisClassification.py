import csv
ext=".csv"
for i in range(1,17):
	filename="results/PredictionOffoodInspectionTest"
	filename=filename+str(i)+ext
	f = open(filename,"r")
	reader=csv.reader(f)
	lines=[]
	for row in reader:
		lines.append(row)
	totalCount=len(lines)
	misClassificationCount=0
	for t in lines:
		if(t[0]!=t[1]):
			misClassificationCount+=1
	rate = float(misClassificationCount)/totalCount		
	print str(rate)
	
