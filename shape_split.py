import math
import sys

def main():

	if len(sys.argv) != 2:
		sys.exit("Usage: python shape_split.py <filename>")
	if not sys.argv[1].endswith(".txt"):
		sys.exit("File Error: Input file must be a text file")
	
	with open(sys.argv[1]) as f:
		shape = f.read()

	print(shape + "\n")
	pieces = break_pieces(shape)

	with open(sys.argv[1][:-4:] + "_split.txt", "w") as f:		
		for piece in pieces:
			print(piece + "\n")
			f.write(piece + "\n\n")
		
# A class to represent vertices of a shape
class Node:

	def __init__(self, pos):
		self.x, self.y = pos 													# (x, y) cooridinate of vertex
		self.up, self.down, self.right, self.left = None, None, None, None 		# The nodes that are adjacent to this node
		self.sub = []  															# Array for any nodes of nested shapes that we will artificially connect to this node 

	def __hash__(self):
		return hash((self.x, self.y))

	def __eq__(self, other):
		return (self.x, self.y) == (other.x, other.y)

	def numNeighbours(self):
			return (self.right is not None) + (self.down is not None) + (self.left is not None) + (self.up is not None)

class Subshape:

	def __init__(self):
		self.nodes = []
		self.minX = math.inf			# The minimum x value of all nodes in this subshape
		self.minY = math.inf			# The minimum y value of all nodes in this subshape
		self.maxX = -math.inf			# The maximum x value of all nodes in this subshape
		self.maxY = -math.inf			# The maximum y value of all nodes in this subshape
		self.nests = []

	def addNode(self, node):			# Adds a node to this subshape and updates the min/max coordinates
		self.nodes.append(node)
		if node.y <= self.minY: self.minY = node.y
		if node.x <= self.minX: self.minX = node.x
		if node.y >= self.maxY: self.maxY = node.y
		if node.x >= self.maxX: self.maxX = node.x

	def updatePositions(self):			# Translates the coordinates of each node to their coordinates relative to the (minX, minY)
		for node in self.nodes:
			node.x -= self.minX
			node.y -= self.minY
		for graph in self.nests:
			for node in graph:
				node.x -= self.minX
				node.y -= self.minY
		self.maxX -= self.minX
		self.minX = 0
		self.maxY -= self.minY
		self.minY = 0

# Debug function used to print the coordinates of a node
def pNode(node):
	if node is not None:
		return (node.x, node.y)
	return None

# Given a graph, reduce the number of shapes this node can be used for 
def reduceGraph(graph, node):
	graph[node] -= 1
	if graph[node] == 0:				# Delete the node if it can't be used for any more shapes
		del graph[node]

# Debug function used to print graphs
def printGraph(graph):
	for node in graph:
		print("Node pos: {}, Node uses: {}".format(pNode(node), graph[node]))
		for sub in node.sub:
			print(pNode(sub), end=' ')
		print('\r')

# Find the node with the smallest y coordinate, then smallest x coordinate in a list of nodes
def minNode(graph):
	m = (math.inf, math.inf)
	n = None
	for node in graph.keys():
		if node.y < m[1] or (node.y == m[1] and node.x <= m[0]):
			m = (node.x, node.y)
			n = node
	return n

# When traversing a subshape or graph border, this is used to determine which node we should go to next 
def traverseNext(curr, direction, rotation): 
	# Direction = the direction we came from {0: right, 1: down, 2: left, 3: up}
	# Rotation = whether we want to traverse clockwise (to find a single shape) or anticlockwise (to find the composition's border) {1: clockwise, -1: anticlockwise}
	neighbours = [curr.right, curr.down, curr.left, curr.up]

	if neighbours[(direction+rotation)%4] is not None:
		newCurr = neighbours[(direction+rotation)%4]
		newDirection = (direction+rotation)%4
	elif neighbours[direction] is not None:
		newCurr = neighbours[direction]
		newDirection = direction
	else:
		newCurr = neighbours[(direction-rotation)%4]
		newDirection = (direction-rotation)%4

	return newCurr, newDirection

def process_input(shape):
	shape = shape.lstrip('\n').splitlines()
	lines = []
	corners = []				# This array keeps track of which nodes we still haven't processed

	for i in range(len(shape)):
		lines.append(shape[i])
		for j in range(len(shape[i])):
			if shape[i][j] not in {'+', '-', '|', '\n', ' '}:
				sys.exit("File Error: Input file has an illegal character")
			if shape[i][j] == '+':
				total = 0
				if i != 0 and j < len(shape[i-1]): total += shape[i-1][j] == '|'
				if i != len(shape) - 1 and j < len(shape[i+1]): total += shape[i+1][j] == '|'
				if j != 0: total += shape[i][j-1] == '-'
				if j != len(shape[i]) - 1: total += shape[i][j+1] == '-'
				if(total < 2): sys.exit("File Error: Input file is an illegal ASCII diagram")
				corners.append(Node((j, i)))
			if shape[i][j] == '-':
				failed = False
				if j != 0: failed = failed or (shape[i][j-1] not in {'+', '-'})
				else: failed = True
				if j != len(shape[i]) - 1: failed = failed or (shape[i][j+1] not in {'+', '-'})
				else: failed = True
				if failed: sys.exit("File Error: Input file is an illegal ASCII diagram")
			if shape[i][j] == '|':
				failed = False
				if i != 0 and j < len(shape[i-1]): failed = failed or (shape[i-1][j] not in {'+', '|'})
				else: failed = True
				if i != len(shape) - 1 and j < len(shape[i+1]): failed = failed or (shape[i+1][j] not in {'+', '|'})
				else: failed = True
				if failed: sys.exit("File Error: Input file is an illegal ASCII diagram")


	return lines, corners

def generate_graphs(lines, corners):
	# The general plan of this function is to place each node into a graph
	# Each graph represents a set of nodes that are directly linked or linked via other adjacent nodes
	# Nested shapes will not be in the same graph as their parent shape as they never connect
	# Here, we will use a dictionary as the graph where the keys are each node belonging to that graph and values are how many shapes that node will be used for
	# Each graph will then be stored in an array containing all graphs
	
	graphs = []

	# This loop will run until all vertices are added to some graph
	while len(corners) != 0:			
		start = corners[0]
		graph = dict()
		visited = [start]
		queue = [start]

		# We perform a breath-first search on the vertices remaining and add each node we find to a graph
		# As we only traverse through connected nodes, we can ensure that no vertices of nested shapes will be added to this graph
		# When we find a node, we initially assume that the number of shapes this node will be a part of is the same as the number of connections it has (this will be corrected later)
		while len(queue) != 0:			
			node = queue.pop(0)
			corners.remove(node)
			graph[node] = 0

			# Searching right of the current node
			if node.x < len(lines[node.y]) - 1:						
				if lines[node.y][node.x + 1] == '-' or lines[node.y][node.x + 1] == '+':
					graph[node] += 1
					for i in range(node.x + 1, len(lines[node.y])):
						if lines[node.y][i] == '+':
							tempNode = Node((i, node.y))
							try:
								tempNode = visited[visited.index(tempNode)]
							except:
								visited.append(tempNode)
								queue.append(tempNode)
							tempNode.left = node
							node.right = tempNode
							break

			# Searching below the current node
			if node.y < len(lines) - 1:								
				if len(lines[node.y + 1]) - 1 >= node.x:
					if lines[node.y + 1][node.x] == '|' or lines[node.y + 1][node.x] == '+':
						graph[node] += 1
						for i in range(node.y + 1, len(lines)):
							if lines[i][node.x] == '+':
								tempNode = Node((node.x, i))
								try:
									tempNode = visited[visited.index(tempNode)]
								except:
									visited.append(tempNode)
									queue.append(tempNode)
								tempNode.up = node
								node.down = tempNode
								break

			# Searching left of the current node
			if node.x > 0:											
				if lines[node.y][node.x - 1] == '-' or lines[node.y][node.x - 1] == '+':
					graph[node] += 1
					for i in range(node.x - 1, -1, -1):
						if lines[node.y][i] == '+':
							tempNode = Node((i, node.y))
							try:
								tempNode = visited[visited.index(tempNode)]
							except:
								visited.append(tempNode)
								queue.append(tempNode)
							tempNode.right = node
							node.left = tempNode
							break

			# Searching above the current node
			if node.y > 0:											
				if len(lines[node.y - 1]) - 1 >= node.x:											
					if lines[node.y - 1][node.x] == '|' or lines[node.y - 1][node.x] == '+':
						graph[node] += 1
						for i in range(node.y - 1, -1, -1):
							if lines[i][node.x] == '+':
								tempNode = Node((node.x, i))
								try:
									tempNode = visited[visited.index(tempNode)]
								except:
									visited.append(tempNode)
									queue.append(tempNode)
								tempNode.down = node
								node.up = tempNode
								break

		# Our initial assumption was that the number of shapes this node will be a part of is the same as the number of connections it has
		# This is not true for the nodes that are on the borders of the current graph
		# Therefore we traverse the border and reduce each border node's dictionary value by one 
		graph[start] -= 1
		curr = start.right
		direction = 0
		while curr != start:
			graph[curr] -= 1
			curr, direction = traverseNext(curr, direction, -1)

		graphs.append(graph)

	return graphs

def add_nest_connections(lines, graphs):
	# Later on, we will need each subshape to know if it has any shapes nested inside them
	# In this section, we artificially create a link between the minNode of any nested graphs with a node in a parent graph
	# This link will allow us to identify the border of a nested shape as a hole in the parent shape when drawing

	for i in range(1, len(graphs), 1):				# We start from graphs[1] because graphs[0] would have been the main graph and so no need to check if it has any parents
		substart = list(graphs[i])[0]				# We identify the minNode of the current graph (We do not need to use the minNode() function as we already know the first element appended was the minNode)
		j = substart.x - 1
		definiteNode = False						# This is a variable to show if our artificial link has been established (It is needed because we cannot break out of multiple loops at once)

		while not definiteNode:
			# Until we find the edge or node of another graph, move left (We know for sure that this won't be a node of this current graph as we start at the minNode)
			while lines[substart.y][j] != '+' and lines[substart.y][j] != '|':
				j -= 1

			k = substart.y

			# If we are on an edge, we want to traverse up and find a node that belongs to the graph we found
			while lines[k][j] != '+':
				k -= 1	

			possibleNode = False					# This variable is to show if we've already concluded whether the found graph is the parent graph or not (Again used because we can only break out of one loop)
			for graph in graphs:
				if not possibleNode:
					for node in graph:
						if (node.x, node.y) == (j, k):

							# It is possible that the graph we found is merely another nested shape within the actual parent graph and so we need to check if the graph we just found is the parent or not
							# In order to do this, we're going to attempt to traverse our graph in a clockwise manner
							# If the graph we found does enclose this shape, the clockwise traversal should traverse the whole parent shape and our traversal would have more clockwise turns than anticlockwise turns
							# If the graph we found is another nested shape, the clockwise traversal will traverse the border of the nested shape (because we started on a right border of the shape) and our traversal will have more anticlockwise turns than clockwise
							leftX = node.x 			# If this graph does not enclose this shape, we'll use this value to restart the left traversal of j
							turns = 0   			# Variable to keep track of how many more clockwise turns we have had compared to anticlockwise turns (will go into negatives if other way around)

							curr = node 			# Curr refers to the current node we are on in our traversal
							direction = 3			# Direction is the direction we just cam from in our traversal {0:right, 1:down, 2:left, 3:left}
							curr, newDirection = traverseNext(curr, direction, 1)
							if (newDirection - direction) % 4 == 1:
								turns += 1
							elif (newDirection - direction) % 4 == 3:
								turns -= 1
							direction = newDirection
							while curr != node:
								newCurr, newDirection = traverseNext(curr, direction, 1)
								if newDirection == 1 and curr.y <= substart.y and newCurr.y >= substart.y and newCurr.x < leftX:
									leftX = newCurr.x
								if (newDirection - direction) % 4 == 1:
									turns += 1
								elif (newDirection - direction) % 4 == 3:
									turns -= 1
								curr = newCurr
								direction = newDirection

							# If we went around a shape that encloses this one
							if turns >= 4:
								definiteNode = True
								if k == substart.y: 			# We traversed onto the node by only going right, hence just append the already existing node
									node.sub.append(substart)
								else:							# We also had to traverse up to find a node in the parent graph, hence we need to create a new node in this graph that is just left of this node
									newNode = Node((j, substart.y))
									newNode.up, newNode.down = node, node.down
									node.down.up = newNode
									newNode.sub.append(substart)
									graph[newNode] = 1
									lines[substart.y] = lines[substart.y][:j:] + '+' + lines[substart.y][j+1::]
							else:
								j = leftX - 1
							possibleNode = True
							break
				else:
					break

def separate_shapes(graphs):
	# To separate shapes, we just need to traverse clockwise on each graph we have and reduce each node's dictionary value as we traverse them
	# By going clockwise, we ensure that each shape we find is a smallest possible decomposition of the current graph
	# Once a node has been traversed as many times as shapes it is part of, it will disappear from the graph, allowing us to move on to shapes with other nodes

	shapes = []

	for graph in graphs:
		
		while len(graph) > 0:
			
			start = minNode(graph)
			curr = start.right								# Represents the current node we are at in the traversal
			direction = 0 									# Represents the direction we came from {0: right, 1: down, 2: left, 3: up}

			# We are going to represent each shape (decomposition) of the graph as a Subshape object
			# The Subshape object stores the nodes that are part of the shape and also the nodes of any shapes nested in this shape
			subshape = Subshape()
			subshape.addNode(Node((start.x, start.y)))

			while curr != start:
				
				# Check to see if the current node is linked to a nested shape
				if direction == 3 or (direction == 0 and curr.down is None):

					# If it is, we traverse the nested shape anticlockwise to find its border and add these border nodes to the Subshape's nests array
					while len(curr.sub) > 0:
						subStart = curr.sub.pop(0)
						nest = [Node((subStart.x, subStart.y))]
						subCurr = subStart.right
						subDirection = 0
						while subCurr != subStart:
							newNode = Node((subCurr.x, subCurr.y))
							if subDirection == 0:
								newNode.left = nest[-1]
								nest[-1].right = newNode
							elif subDirection == 1:
								newNode.up = nest[-1]
								nest[-1].down = newNode
							elif subDirection == 2:
								newNode.right = nest[-1]
								nest[-1].left = newNode
							elif subDirection == 3:
								newNode.down = nest[-1]
								nest[-1].up = newNode
							nest.append(newNode)
							subCurr, subDirection = traverseNext(subCurr, subDirection, -1)

						nest[0].down = nest[-1]
						nest[-1].up = nest[0]
						subshape.nests.append(nest)	


				reduceGraph(graph, curr)
				
				newNode = Node((curr.x, curr.y))			# When we add new nodes to the Subshape, we need to ensure they are completely new nodes and not pointers to existing nodes
															# This allows us to perform separate operations to nodes with the same coordinates if they are present in multiple Subshapes
				if direction == 0: 
					newNode.left = subshape.nodes[-1]
					subshape.nodes[-1].right = newNode
				elif direction == 1: 
					newNode.up = subshape.nodes[-1]
					subshape.nodes[-1].down = newNode
				elif direction == 2: 
					newNode.right = subshape.nodes[-1]
					subshape.nodes[-1].left = newNode
				elif direction == 3: 
					newNode.down = subshape.nodes[-1]
					subshape.nodes[-1].up = newNode
				
				curr, direction = traverseNext(curr, direction, 1)

				subshape.addNode(newNode)

			reduceGraph(graph, start)
			subshape.nodes[0].down = subshape.nodes[-1]
			subshape.nodes[-1].up = subshape.nodes[0]
			shapes.append(subshape)

	return shapes

def draw_shapes(shapes):
	# Finally we can draw each Subshape we have and use the nests array to draw holes in these shapes

	answer = []
	for subshape in shapes:

		subshape.updatePositions()

		# Create the initial string array with just spaces
		lines = []
		for i in range(subshape.maxY + 1):
			lines.append([])
			for j in range(subshape.maxX + 1):
				lines[i].append(' ')

		subshape.nests.insert(0, subshape.nodes)			# We're going to move the array storing the nodes of the subshape to the nests array to make looping simpler

		for nodeList in subshape.nests:

			size = len(nodeList)
			for n in range(size):
				node = nodeList[n]
				nodeType = '+'								# For now we assume that this node will also be a vertex of the shape and make it a '+'
															# It is possible that this node was merely a vertex of another shape joining on to an edge of this shape and so when drawing this shape, this node only needs to be an edge
															# We check for this below
				
				# Whatever the next node in the shape is, fill our corresponding '-' or '|' to reach it
				if node.right is nodeList[(n+1)%size]:
					for i in range(node.x+1, node.right.x, 1):
						lines[node.y][i] = '-'
					if node.left is not None: nodeType = '-'
				if node.down is nodeList[(n+1)%size]:
					for i in range(node.y+1, node.down.y, 1):
						lines[i][node.x] = '|'
					if node.up is not None: nodeType = '|'
				if node.left is nodeList[(n+1)%size]:
					for i in range(node.x-1, node.left.x, -1):
						lines[node.y][i] = '-'
					if node.right is not None: nodeType = '-'
				if node.up is nodeList[(n+1)%size]:
					for i in range(node.y-1, node.up.y, -1):
						lines[i][node.x] = '|'
					if node.down is not None: nodeType = '|'

				lines[node.y][node.x] = nodeType
		
		# Format the answer
		string = ""
		for i in range(len(lines)):
			if i == len(lines) - 1:
				string += "".join(lines[i]).rstrip()
			else:
				string += "".join(lines[i]).rstrip() + '\n'
		answer.append(string)
		
	return answer

def break_pieces(shape):

	# ---- PROCESS INPUT ----

	lines, corners = process_input(shape)
	
	# ---- GENERATE GRAPHS ---- 
	
	graphs = generate_graphs(lines, corners)

	# ---- ADD NESTED SHAPE CONNECTIONS ----
	
	add_nest_connections(lines, graphs)

	# ---- SEPARATE SHAPES ----
	
	shapes = separate_shapes(graphs)

	# ---- DRAW SHAPES -----
	
	return draw_shapes(shapes)

if __name__ == "__main__":
	main()

#shape = '\n+------------+\n|            |\n|            |\n|            |\n+------+-----+\n|      |     |\n|      |     |\n+------+-----+'
#shape = '\n+------+\n|      |\n| +-+  |\n| | |  +-+\n+-+ +--+ |\n| |      |\n+-+------+'
#shape = '\n+-------------------+--+\n|                   |  |\n|                   |  |\n|  +----------------+  |\n|  |                   |\n|  |                   |\n+--+-------------------+'
#shape = '\n         +------------+--+      +--+\n         |            |  |      |  |\n         | +-------+  |  |      |  |\n         | |       |  |  +------+  |\n         | |       |  |            |\n         | |       |  |    +-------+\n         | +-------+  |    |        \n +-------+            |    |        \n |       |            |    +-------+\n |       |            |            |\n +-------+            |            |\n         |            |            |\n    +----+---+--+-----+------------+\n    |    |   |  |     |            |\n    |    |   |  +-----+------------+\n    |    |   |                     |\n    +----+---+---------------------+\n    |    |                         |\n    |    | +----+                  |\n+---+    | |    |     +------------+\n|        | |    |     |             \n+--------+-+    +-----+             '
#shape = '\n                 \n   +-----+       \n   |     |       \n   |     |       \n   +-----+-----+ \n         |     | \n         |     | \n         +-----+ '
#shape = '\n+-------+ +----------+\n|       | |          |\n| +-+ +-+ +-+    +-+ |\n+-+ | |     |  +-+ +-+\n    | +-----+--+\n+-+ |          +-+ +-+\n| +-+  +----+    | | |\n| |    |    |    +-+ |\n| +----++ +-+        |\n|       | |          |\n+-------+ +----------+'
#shape = '\n+--------+\n|        |\n|  +--+  |\n|  |  |  |\n|  +--+  |\n|        |\n|  +--+  |\n|  |  |  |\n|  +--+  |\n|        |\n+--------+'
#shape = '\n+-----+\n|     |\n| +-+ |\n| | | |\n+-+-+ |\n  |   |\n  +---+'
#shape = '\n+--------+\n|        |\n| +--+-+ |\n| |  | | |\n| +--+-+ |\n|        |\n| +----+ |\n| |    | |\n| | ++ | |\n| | ++ | |\n| |    | |\n| +----+ |\n|        |\n+--------+' 
#shape = '\n+--------------------------------------------------------+\n|                                                        |\n|   +---+    +---+    +---+    +---+    +---+    +---+   |\n|   |   |    |   |    |   |    |   |    |   |    |   |   |\n|   +---+    +---+    +---+    +---+    +---+    +---+   |\n|                                                        |\n+--------------------------------------------------------+'
#shape = '\n+---------------------------------------------------------------------------------+\n|                                                                                 |\n|  +---------------------------------------------------------------------------+  |\n|  |                                                                           |  |\n|  |  +--------+    +-+-+---+         +-+         +---------+     +--+--+--+   |  |\n|  |  |        |    | | |   |       +-+ +-+       |         |     |  |  |  |   |  |\n|  |  |  +--+  |    | | +-+ |     +-+     +-+     ++       ++     +--+--+--+   |  |\n|  |  |  |  |  |    | |   | |     |   +-+   |      ++     ++      |  |  |  |   |  |\n|  |  |  +--+  |    | +---+-+     ++  | |  ++       ++   ++       +--+--+--+   |  |\n|  |  |        |    |       |      |  | |  |         ++ ++        |  |  |  |   |  |\n|  |  +--------+    +-------+      +--+-+--+          +-+         +--+--+--+   |  |\n|  |                                                                           |  |\n|  +---------------------------------------------------------------------------+  |\n|                                                                                 |\n+---------------------------------------------------------------------------------+'
#shape = '\n+--+   +--+---------+\n|  |   |  |         |\n+--+---+--+   +-+   |\n   |   |  |   | |   |\n  ++  +++ +-+ +-+ +-+\n  |   | |   |     |\n+-+ +-+-+-+-+-----+-+\n|   |   | | |     | |\n+---+   +-+-+     +-+'
#shape = '\n+---------+ +-------+\n|         | |       |\n|         +-+  +--+ |\n|         |    |  | |\n|         | +--+--+ |\n|         | |  |    |\n|         | +--+  +-+\n|         |       | |\n+---------+---+---+-+\n          |   |   | |\n          +-+ +-+-+-+\n            |   | |\n            ++  +++\n             |   |\n          +--+---+--+\n          |  |   |  |\n          +--+   +--+'


"""
pieces = break_pieces(shape)

for piece in pieces:
	print(piece)
"""