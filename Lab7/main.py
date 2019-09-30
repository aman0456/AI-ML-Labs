from sudoku import SudokuSearchProblem, parse_grid, assign
from maps import MapSearchProblem	
from search import sudokuDepthFirstSearch, AStar_search, heuristic, nullHeuristic
import time, sys
import osmnx as ox
################ Utilities ################
def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]
    
digits   = '123456789'
rows     = 'ABCDEFGHI'
cols     = digits
squares  = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')])
units = dict((s, [u for u in unitlist if s in u])
             for s in squares)
peers = dict((s, set(sum(units[s],[]))-set([s]))
             for s in squares)

def from_file(filename, sep='\n'):
	"Parse a file into a list of strings, separated by sep."
	return file(filename).read().strip().split(sep)

def solve(grid): return search_DFS(parse_grid(grid))

def search_DFS(values):
	"Using a-star search and propagation."
	
	if values is False:
		return False, 0 ## Failed earlier
	
	if all(len(values[s]) == 1 for s in squares):
		return values, 0 ## Solved!
	
	prob = SudokuSearchProblem(values)    
	vals = sudokuDepthFirstSearch(prob)
	
	n_expanded = prob.nodes_expanded
	return vals, n_expanded

def solve_all(grids, name='', showif=0.0):
	"""Attempt to solve a sequence of grids. Report results.
	"""

	def time_solve(grid):
		start = time.clock()
		n_expanded = 0
		values, n_expanded = solve(grid)
		t = time.clock()-start
		return (t, n_expanded, solved(values))

	times, ns_expanded, results = zip(*[time_solve(grid) for grid in grids])
	N = len(grids)
	if N > 1:
		print "Solved %d of %d of the %s puzzles (avg %.2f secs (%d Hz), max %.2f secs)." % (
			sum(results), N, name, sum(times)/N, N/sum(times), max(times))
		print "Nodes expanded = (avg %.2f, max %.2f )." % (
			sum(ns_expanded) / N, max(ns_expanded))

	return times, ns_expanded, results

def solved(values):
	"A puzzle is solved if each unit is a permutation of the digits 1 to 9."
	def unitsolved(unit): return set(values[s] for s in unit) == set(digits)
	return values is not False and all(unitsolved(unit) for unit in unitlist)

def task1():
		
	grids = [0]*3
	grids[0] = '3.6.7...........518.........1.4.5...7.....6.....2......2.....4.....8.3.....5.....'
	grids[1] = '85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4.'
	grids[2] = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'
	corner_case = {'G7': '1', 'G6': '3', 'G5': '5', 'G4': '6', 'G3': '2', 'G2': '9', 'G1': '8', 'G9': '4', 'G8': '7', 'C9': '2', 'C8': '1', 'C3': '5', 'C2': '8', 'C1': '6', 'C7': '169', 'C6': '4', 'C5': '3', 'C4': '7', 'A9': '5', 'A8': '3', 'F1': '', 'F2': '4', 'F3': '6', 'F4': '5', 'F5': '1', 'F6': '2', 'F7': '3', 'F8': '9', 'F9': '8', 'B4': '1', 'B5': '6', 'B6': '5', 'B7': '', 'B1': '2', 'B2': '3', 'B3': '9', 'B8': '4', 'B9': '7', 'I9': '3', 'I8': '5', 'I1': '1', 'I3': '4', 'I2': '6', 'I5': '7', 'I4': '8', 'I7': '2', 'I6': '9', 'A1': '4', 'A3': '7', 'A2': '1', 'E9': '7', 'A4': '9', 'A7': '8', 'A6': '', 'E5': '8', 'E4': '3', 'E7': '4', 'E6': '6', 'E1': '9', 'E3': '1', 'E2': '5', 'E8': '2', 'A5': '2', 'H8': '8', 'H9': '6', 'H2': '7', 'H3': '3', 'H1': '5', 'H6': '1', 'H7': '9', 'H4': '2', 'H5': '4', 'D8': '6', 'D9': '1', 'D6': '7', 'D7': '5', 'D4': '4', 'D5': '9', 'D2': '2', 'D3': '8', 'D1': ''}
	marks_for_parts = [0,0,0]
	for grid in grids:
		prob = SudokuSearchProblem(parse_grid(grid))
		
		# Test Get Start State Function
		if prob.getStartState() == parse_grid(grid): 
			marks_for_parts[0] += 1.0 / 6

	# Test is Goal State Function
	if not prob.isGoalState(parse_grid(grids[1])):
		marks_for_parts[1] += 1.0/4
	if prob.isGoalState(parse_grid(grids[2])):
		marks_for_parts[1] += 1.0/4

	# Test the getSuccessors Function
	curr_vals = parse_grid(grids[1])
	succs = prob.getSuccessors(curr_vals)
	if len(succs) > 1 and all(succ[0] == assign(curr_vals.copy(), succ[1][0], succ[1][1]) for succ in succs):
		marks_for_parts[2] += 1
	
	if len(prob.getSuccessors(corner_case)) == 0:
		marks_for_parts[2] += 1
	marks_for_parts_str = [str(x) for x in marks_for_parts]
	return ' + '.join(marks_for_parts_str) + " = " + str(sum(marks_for_parts)), 3

def task2():
	marks_for_parts = [0] * 2
	
	times, ns_expanded, results = solve_all(from_file("data/sudoku/top95.txt"), "top95", None)
	if all(results) and sum(ns_expanded) / len(ns_expanded) < 35:
		marks_for_parts[0] += 1

	times, ns_expanded, results = solve_all(from_file("data/sudoku/hardest.txt"), "hardest", None)
	if all(results) and sum(ns_expanded) / len(ns_expanded) < 15:
		marks_for_parts[1] += 1

	marks_for_parts_str = [str(x) for x in marks_for_parts]
	return ' + '.join(marks_for_parts_str) + " = " + str(sum(marks_for_parts)), 2	

## Helper Functions for Section 2
def get_nearest_node(G, point):
	nodes = list(G.nodes)
	distances = [((G.node[node]['x']-point['x'])**2 + (G.node[node]['y']-point['y'])**2)**0.5 for node in nodes]
	index, element = min(enumerate(distances), key=lambda x: x[1])
	return nodes[index]

def get_problem(problem='iit'):
	assert problem in ['iit', 'mumbai', 'random1', 'random2', 'random3']

	G = ox.load_graphml('maps/%s.graphml' % problem)

	if problem == 'iit':
		start = {'x':72.9104,'y':19.1362}
		end = {'x':72.9163,'y':19.1295}
		origin_node = get_nearest_node(G, start)
		final_node = get_nearest_node(G, end)
	elif problem == 'mumbai':
		start = {'x':72.9074, 'y':19.1354}
		end = {'x':72.8171, 'y':18.9512}
		origin_node = get_nearest_node(G, start)
		final_node = get_nearest_node(G, end)
	else:
		origin_node = list(G.nodes())[4]
		final_node = list(G.node())[14]

	return G, origin_node, final_node

def get_distance(G, route):
	distance = 0
	for i, x in enumerate(route[:-1]):
		distance += G[x][route[i+1]][0]['length']

	return distance

def task4():

	grids = [0]*3
	grids[0] = get_problem('random1')
	grids[1] = get_problem('random2')
	grids[2] = get_problem('random3')

	marks_for_parts = [0,0,0]

	successors = [set([65303556, 65314173, 65354431]), set([4850315874]), set([53046249, 53124803, 53125997])]
	getsucnode = [65327163, 4850315869, 53127240]
	for i, grid in enumerate(grids):
		prob = MapSearchProblem(*grid)
		
		# Test Get Start State Function
		if prob.getStartState() == grid[1]: 
			marks_for_parts[0] += 1.0 / 6
		# Test Is Goal State Function
		if prob.isGoalState(grid[2]):
			marks_for_parts[1] += 1.0 / 12
		if not prob.isGoalState(list(grid[0].nodes())[3]):
			marks_for_parts[1] += 1.0 / 12

		# Test Get Successors function
		succs = set([x[0] for x in prob.getSuccessors(getsucnode[i])])
		if succs == successors[i]:
			marks_for_parts[2] += 2.0 / 3

	marks_for_parts_str = [str(x) for x in marks_for_parts]
	return ' + '.join(marks_for_parts_str) + " = " + str(sum(marks_for_parts)), 3

def task5(to_show):

	marks_for_part = 0
	G_iit, origin_node, final_node = get_problem('iit')
	actual_route = [668478124, 668672699, 4348259295, 668672574, 245674992, 668672700, 3215340463, 667357325, 5366445105]

	problem = MapSearchProblem(G_iit, origin_node, final_node)
	route_astar_iit = AStar_search(problem, nullHeuristic)
	if route_astar_iit == actual_route:
		marks_for_part += 1

	print "Number of nodes expanded for IIT: %d" % problem.nodes_expanded

	G_mumbai, origin_node, final_node = get_problem('mumbai')
	actual_route = [5280438428, 4348259324, 5280437314, 5280437313, 668672574, 245674992, 668672700, 3215340463, 667357325, 668699249, 667357311, 668660552, 1879817556, 245673500, 3708892000, 667395479, 667395482, 667395474, 667395468, 2246662857, 2246662767, 1855812188, 667395460, 2055634531, 4759328799, 4759328802, 1366901332, 4759328760, 5456498950, 5456509646, 5456509636, 5456509634, 2245075822, 1366901213, 1366901212, 2245075777, 1366901170, 5456500624, 5456509624, 2245075731, 1366901143, 2245075698, 1366901132, 2244962270, 2244962262, 2244962226, 2244962231, 2244962200, 2244962171, 1366900900, 4366100155, 2246661216, 2246661194, 3245621937, 2246661136, 2246661130, 2246661132, 2246661074, 623889405, 5477703669, 5477703667, 5477703671, 5758624138, 2622177356, 2244961487, 1644086520, 623889400, 4264598038, 4264598040, 4264598033, 4264598029, 4264598395, 1640287824, 1640287806, 1454672572, 245668721, 1454673995, 1643047708, 245668490, 2246472636, 1643047703, 2246472590, 2246472543, 2246472486, 2246472465, 1643035298, 1643035294, 2246472375, 1643031073, 245667692, 2146269450, 2163935132, 2146273510, 1232644012, 4264598321, 2146273513, 5913464524, 2439252028, 5865538271, 1366938254, 5865538272, 1232644986, 1642951891, 1642951846, 1642951818, 245664866, 1642951755, 1642951748, 1642954393, 1179340118, 1179340221, 245664141, 245663296, 1179314459, 245663089, 4362786418, 1643134423, 2258304514, 2258304493, 2258304484, 1179089888, 1179257896, 1179273129, 245660324, 2258304455, 245660183, 1179221113, 1179221116, 1179221118, 5921458181, 2258304438, 245659911, 245659814, 1165320838, 5848596597, 245659701, 2258304421, 245659569, 245659514, 245659478, 245659392, 2258304411, 674906578, 2258304349, 2258304338, 5637109060, 1375482657, 245657044, 245657005, 620382315, 2258304310, 245656839, 4443807473, 861155618, 861155314, 861155306, 2246157812, 245656050, 861155655, 861190736, 620393213, 245655857, 245655720, 245655652, 2246157809, 245655525, 245655412, 245655339, 620403338, 245655305, 2184321321, 245655223, 5510030709, 245655072, 620408597, 620404556, 245654969, 245654863, 2248513091, 2244632754, 1643210517, 245654715, 1643210514, 485953796, 485953814, 485953784, 356168287, 1643210508, 6215526121, 1127766407, 4568979756, 4568979754]
	problem = MapSearchProblem(G_mumbai, origin_node, final_node)
	route_astar_mumbai = AStar_search(problem, nullHeuristic)
	if route_astar_mumbai == actual_route:
		marks_for_part += 1
	print "Number of nodes expanded for IIT: %d" % problem.nodes_expanded

	if to_show:
		try:
			ox.plot_graph_route(G_iit, route_astar_iit, node_size=1)
		except:
			print "The route returned is not possible for IIT"
		try:
			ox.plot_graph_route(G_mumbai, route_astar_mumbai, node_size=1)
		except:
			print "The route returned is not possible for Mumbai"	

	return marks_for_part, 2

def task6(to_show):

	marks_for_part = 0

	G_iit, origin_node, final_node = get_problem('iit')
	actual_route = [668478124, 668672699, 4348259295, 668672574, 245674992, 668672700, 3215340463, 667357325, 5366445105]
	problem = MapSearchProblem(G_iit, origin_node, final_node)
	route_astar_iit = AStar_search(problem, heuristic)

	if route_astar_iit == actual_route and problem.nodes_expanded < 25:
		marks_for_part += 1 
	print "Number of nodes expanded for IIT: %d" % problem.nodes_expanded

	G_mumbai, origin_node, final_node = get_problem('mumbai')
	actual_route = [5280438428, 4348259324, 5280437314, 5280437313, 668672574, 245674992, 668672700, 3215340463, 667357325, 668699249, 667357311, 668660552, 1879817556, 245673500, 3708892000, 667395479, 667395482, 667395474, 667395468, 2246662857, 2246662767, 1855812188, 667395460, 2055634531, 4759328799, 4759328802, 1366901332, 4759328760, 5456498950, 5456509646, 5456509636, 5456509634, 2245075822, 1366901213, 1366901212, 2245075777, 1366901170, 5456500624, 5456509624, 2245075731, 1366901143, 2245075698, 1366901132, 2244962270, 2244962262, 2244962226, 2244962231, 2244962200, 2244962171, 1366900900, 4366100155, 2246661216, 2246661194, 3245621937, 2246661136, 2246661130, 2246661132, 2246661074, 623889405, 5477703669, 5477703667, 5477703671, 5758624138, 2622177356, 2244961487, 1644086520, 623889400, 4264598038, 4264598040, 4264598033, 4264598029, 4264598395, 1640287824, 1640287806, 1454672572, 245668721, 1454673995, 1643047708, 245668490, 2246472636, 1643047703, 2246472590, 2246472543, 2246472486, 2246472465, 1643035298, 1643035294, 2246472375, 1643031073, 245667692, 2146269450, 2163935132, 2146273510, 1232644012, 4264598321, 2146273513, 5913464524, 2439252028, 5865538271, 1366938254, 5865538272, 1232644986, 1642951891, 1642951846, 1642951818, 245664866, 1642951755, 1642951748, 1642954393, 1179340118, 1179340221, 245664141, 245663296, 1179314459, 245663089, 4362786418, 1643134423, 2258304514, 2258304493, 2258304484, 1179089888, 1179257896, 1179273129, 245660324, 2258304455, 245660183, 1179221113, 1179221116, 1179221118, 5921458181, 2258304438, 245659911, 245659814, 1165320838, 5848596597, 245659701, 2258304421, 245659569, 245659514, 245659478, 245659392, 2258304411, 674906578, 2258304349, 2258304338, 5637109060, 1375482657, 245657044, 245657005, 620382315, 2258304310, 245656839, 4443807473, 861155618, 861155314, 861155306, 2246157812, 245656050, 861155655, 861190736, 620393213, 245655857, 245655720, 245655652, 2246157809, 245655525, 245655412, 245655339, 620403338, 245655305, 2184321321, 245655223, 5510030709, 245655072, 620408597, 620404556, 245654969, 245654863, 2248513091, 2244632754, 1643210517, 245654715, 1643210514, 485953796, 485953814, 485953784, 356168287, 1643210508, 6215526121, 1127766407, 4568979756, 4568979754]
	problem = MapSearchProblem(G_mumbai, origin_node, final_node)
	route_astar_mumbai = AStar_search(problem, heuristic)

	if route_astar_mumbai == actual_route and problem.nodes_expanded < 15000:
		marks_for_part += 1
	print "Number of nodes expanded for Mumbai: %d" % problem.nodes_expanded

	if to_show:
		try:
			ox.plot_graph_route(G_iit, route_astar_iit, node_size=1)
		except:
			print "The route returned is not possible for IIT"
		try:
			ox.plot_graph_route(G_mumbai, route_astar_mumbai, node_size=1)
		except:
			print "The route returned is not possible for Mumbai"	
	
	return marks_for_part, 2

def runTask(task, to_show):
	print "Grading task "+str(task) 
	if task == 3 or task == 7:
		print "This is a manually graded task, write your answers in the answers.txt file"
	else :
		if task == 1:
			marks, totmarks = task1()
		
		elif task == 2:
			marks, totmarks = task2()
		
		elif task == 4:
			marks, totmarks = task4()
		
		elif task == 5:
			marks, totmarks = task5(to_show)

		elif task == 6:
			marks, totmarks = task6(to_show)
		
		print "Received Marks : " + str(marks) +"/" +str(totmarks)
	print "--------------------------------------------------------"		

from optparse import OptionParser

if __name__ == "__main__":

	parser = OptionParser("")
	parser.add_option('-t', '--task', help='The task to autograde', choices=['1', '2', '3', '4', '5', '6', '7'], default=None)
	parser.add_option('-s', '--show', help='To show graph', action='store_true', default=False)

	options, otherjunk = parser.parse_args(sys.argv[1:])
	if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))

	task_list = [1, 2, 3, 4, 5, 6, 7]
	if options.task:
		runTask(int(options.task), options.show)
	else:
		for task in task_list:
			try :
				runTask(task, options.show)
			except:
				print "Task" +str(task)	+" exited without running to completion"
				print "Received Marks : " + str(0)
				print "--------------------------------------------------------"

	# call function accordingly
