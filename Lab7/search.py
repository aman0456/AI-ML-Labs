import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Return the final values dictionary, i.e. the values dictionary which is the goal state  
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored/visited set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        You may or may not use this
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return ''.join(modl)

    mystack = util.Stack()
    visited = set()
    mystack.push(problem.getStartState())
    visited.add(convertStateToHash(problem.getStartState()))
    while not mystack.isEmpty():
        topelem = mystack.pop()
        if problem.isGoalState(topelem):
            return topelem
        successors = problem.getSuccessors(topelem)
        for nextelement in successors:
            nextelem = nextelement[0]
            if convertStateToHash(nextelem) in visited:
                continue
            visited.add(convertStateToHash(nextelem))
            mystack.push(nextelem)
    assert(False)



    ## YOUR CODE HERE
    util.raiseNotDefined()

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.

    """
        Takes the state and the problem as input and returns the heuristic for the state
        Returns a real number(Float)
    """
    goalState = problem.end_node
    goalState = problem.G.node[goalState]
    stateState = problem.G.node[state]
    goalP = ((goalState['x'],0,0),(goalState['y'],0,0))
    stateP = ((stateState['x'],0,0),(stateState['y'],0,0))
    return util.points2distance(goalP, stateP)
    util.raiseNotDefined()

def AStar_search(problem, heuristic=nullHeuristic):

    """
        Search the node that has the lowest combined cost and heuristic first.
        Return the route as a list of nodes(Int) iterated through starting from the first to the final.
    """
    pq = util.PriorityQueue()
    parent = {}
    st = problem.getStartState()
    pq.update((st, 0.0, -1), heuristic(st, problem))
    vis = set()
    while not pq.isEmpty():
        curnode, curdist, par = pq.pop()
        if curnode in vis:
            continue
        vis.add(curnode)
        parent[curnode] = par
        if problem.isGoalState(curnode):
            parPath = []
            while(curnode != -1):
                parPath.append(curnode)
                curnode = parent[curnode]
            parPath.reverse()
            return parPath
        succs = problem.getSuccessors(curnode)
        for neigh in succs:
            nextnode, _, pathLength = neigh
            if nextnode in vis:
                continue
            nextdist = pathLength + curdist;
            pq.update((nextnode, nextdist, curnode), nextdist+heuristic(nextnode, problem))
    assert(False)
    util.raiseNotDefined()
