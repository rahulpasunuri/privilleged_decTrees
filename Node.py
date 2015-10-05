class Node:
	def __init__(self, col = -1, leafValues = None, leftBranch = None, rightBranch = None, criteria=None):
		'''
		Explanation of values : 
			col -> The column number of the feature used for split
			leafValue -> Value at the leaf Node, it will be None for all nodes except leaves (It also indicates end of tree)
			leftBranch -> This is the branch contains the nodes that fail the test criteria
			rightBranch -> This is the branch contains the nodes that pass the test criteria
			criteria -> The test condition that is used to decide which samples go the left or right branches
		'''
		
		self.col = col
		self.leafValues = leafValues
		self.leftBranch = leftBranch
		self.rightBranch = rightBranch
		self.criteria = criteria


