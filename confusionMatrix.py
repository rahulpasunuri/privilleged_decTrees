import csv
ext=".csv"
for i in range(1,3):
	filename="results/PredictionOffoodInspectionTest"
	filename=filename+str(i)+ext
	f = open(filename,"r")
	
	reader=csv.reader(f)
	lines=[]
	for row in reader:
		lines.append(row)

	distinctClasses={}
	reverseIndex={}
	totalCount=len(lines)
	t=[]
	liTuples=[]
	count=0
	for t in lines:
		liTuples.append(t)
		if t[1] not in distinctClasses:
			distinctClasses[t[1]]=count
			reverseIndex[count]=t[1]
			count+=1
	numClasses=len(distinctClasses.keys())


	#create the matrix
	cfm1=[0]
	cfm2=cfm1*numClasses
	cfm3=[cfm2]
	cfm=cfm3*numClasses

	cfm = [[0 for x in range(0,numClasses)] for x in range(0,numClasses)]

	for t in liTuples:
		#left entry in csv is predicted label
		#right entry in csv is actual label
		rowIndex=distinctClasses[t[1]]
		columnIndex=distinctClasses[t[0]]		
		cfm[rowIndex][columnIndex]=cfm[rowIndex][columnIndex]+1	

	print "Confusion Matrix with depth - "+str(i)+":"	

	headers = [reverseIndex[j] for j in range(0,numClasses)] 

	print "	"+"	".join(headers)
	count=0
	for row in cfm:
		pString=""	
		for j in row:
			pString+="	"+str(j)
		print headers[count]+pString
		count+=1
	print
	
	
	
	
	
					
