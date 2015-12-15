import util
import time
## Abstract Search Classes

class SearchProblem:
  """
  Abstract SearchProblem class. Your classes
  should inherit from this class and override 
  all the methods below
  """
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze"""
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]


def getActions(path):

  startTime = time.time()

  index = len(path) - 1
  actions = []

  if(index < 0):
    print "path is empty!"
    return null
  
  (parentState, state, action) = path[index]
  index -= 1
  
  while index >= 0:
    (grandparentState, state, action) = path[index]
    if (parentState == state) and not(action == None):
      parentState = grandparentState
    else:
      path[index] = None
    index -= 1

  
  for index in range(0, len(path)):
    if not (path[index] == None):
      actions.append(path[index][2])
    
  return actions

    
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first. [p 74].

    Your search algorithm needs to return a list of actions that reaches
    the goal.
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()
    fringe.push( (problem.getStartState(), [], []) )
    while not fringe.isEmpty():
        node, actions, visited = fringe.pop()

        for coord, direction, steps in problem.getSuccessors(node):
            if not coord in visited:
                if problem.isGoalState(coord):
                    return actions + [direction]
                fringe.push((coord, actions + [direction], visited + [node] ))

    return []
    closed = set()
    fringe = util.Stack()
    fringe.push(makeNode(problem.getStartState()))

    while True:
        if fringe.isEmpty():
            return []

        node = fringe.pop()

        if problem.isGoalState(node.getState()):
            return node.getActions()
        if node.getState() not in closed:
            closed.add(node.getState())
            insertAll(fringe, expand(node, problem))
def makeNode(state):
    return Node(state)

def expand(node, problem):
    parentNode = node
    childList = []
    #print node.getActions()
    successors = problem.getSuccessors(node.getState())
    #print successors
    for i in range(len(successors)):
        successor = successors[i]
        child = makeNode(successor[0])
        child.setActionList(parentNode.getActions())
        child.addAction(successor[1])
        child.setCumulativeCost(parentNode.getCumulativeCost() + successor[2])
        childList.append(child)
    return childList


"""def insertAll(stack, nodeList):
    for node in nodeList:
        stack.push(node)

def insertAllInPriorityQueue(queue, nodeList):
    for node in nodeList:
        queue.push(node, node.getDepth())

def insertAllWithCost(queue, nodeList):
    for node in nodeList:
        queue.push(node, node.getCumulativeCost())"""

def insertAllWithCost(queue, nodeList):
    for node in nodeList:
        queue.push(node, node.getCumulativeCost())

def insertAllWithCostAndHeuristic(queue, nodeList, heuristic, problem):
    for node in nodeList:
        queue.push(node, node.getCumulativeCost() + heuristic(node.getState(), problem))

class Node:

    def __init__(self, state):
        self.actionList = []
        self.state = state
        self.cumulativeCost = 0

    def getState(self):
        return self.state

    def getActions(self):
        return self.actionList

    def setActionList(self, actions):
        self.actionList = actions[:]

    def addAction(self, action):
        self.actionList.append(action)

    def getDepth(self):
        return len(self.actionList)

    def getCumulativeCost(self):
        return self.cumulativeCost

    def setCumulativeCost(self, cost):
        self.cumulativeCost = cost

def breadthFirstSearch(problem):
    "Search the shallowest nodes in the search tree first. [p 74]"
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    fringe.push( (problem.getStartState(), []) )

    visited = []
    while not fringe.isEmpty():
        node, actions = fringe.pop()

        for coord, direction, steps in problem.getSuccessors(node):
            if not coord in visited:
                if problem.isGoalState(coord):
                    return actions + [direction]
                fringe.push((coord, actions + [direction]))
                visited.append(coord)

    return []

# get shortest distance using bfs
def getShortestDist(start, target, walls):
    "Find shortest distances to other dots"
    fringe = util.Queue()
    fringe.push( (start, []) )

    visited = []
    while not fringe.isEmpty():
        node, actions = fringe.pop()

        for coord, direction, steps in getSuccessors(node, walls):
            if not coord in visited:
                if coord[0] == target[0] and coord[1] == target[1]:
                    return len(actions + [direction])
                fringe.push((coord, actions + [direction]))
                visited.append(coord)

    return None

def getSuccessors(state, walls):
    from game import Directions, Actions
    successors = []
    for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      dx, dy = Actions.directionToVector(direction)
      nextx, nexty = int(x + dx), int(y + dy)
      if not walls[nextx][nexty]:
        successors.append( ( (nextx, nexty), direction, 1) )
    return successors
      
def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    # fringe = util.PriorityQueue()
    # fringe.push((problem.getStartState(), ()), 0)
    # explored = []
    #
    # while not fringe.isEmpty():
    #     node, actions = fringe.pop()
    #
    #     if problem.isGoalState(node):
    #         return actions
    #
    #     explored.append(node)
    #
    #     for coord, direction, steps in problem.getSuccessors(node):
    #         if not coord in explored:
    #             new_actions = actions + (direction, )
    #             fringe.push((coord, new_actions), problem.getCostOfActions(new_actions))
    #
    # return []

    closed = set()
    fringe = util.PriorityQueue()
    node = makeNode(problem.getStartState())
    node.setCumulativeCost(0)
    fringe.push(node, node.getCumulativeCost())

    while True:
        if fringe.isEmpty():
            return []

        node = fringe.pop()

        if problem.isGoalState(node.getState()):
            return node.getActions()
        if node.getState() not in closed:
            closed.add(node.getState())
            insertAllWithCost(fringe, expand(node, problem))

  
def nullHeuristic(state):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided searchProblem.  This one is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    """closedset = []
    fringe = util.PriorityQueue()
    start = problem.getStartState()
    fringe.push((start, ()), heuristic(start, problem))

    while not fringe.isEmpty():
        node, actions = fringe.pop()

        if problem.isGoalState(node):
            return actions
        closedset.append(node)

        for coord, direction, cost in problem.getSuccessors(node):
            if not coord in closedset:
                new_actions = actions + (direction, )
                score = problem.getCostOfActions(new_actions) + heuristic(coord, problem)
                fringe.push((coord, new_actions), score)

    return []"""
    closed = set()
    fringe = util.PriorityQueue()
    node = makeNode(problem.getStartState())
    node.setCumulativeCost(0)
    fringe.push(node, node.getCumulativeCost() + heuristic(node.getState(), problem))

    while True:
        if fringe.isEmpty():
            return []

        node = fringe.pop()

        if problem.isGoalState(node.getState()):
            return node.getActions()
        if node.getState() not in closed:
            closed.add(node.getState())
            insertAllWithCostAndHeuristic(fringe, expand(node, problem), heuristic, problem)




def greedySearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest heuristic first."
    "*** YOUR CODE HERE ***"
    closedset = []
    fringe = util.PriorityQueue()
    start = problem.getStartState()
    fringe.push((start, ()), heuristic(start))

    while not fringe.isEmpty():
        node, actions = fringe.pop()

        if problem.isGoalState(node):
            return actions

        closedset.append(node)

        for coord, direction, cost in problem.getSuccessors(node):
            if not coord in closedset:
                new_actions = actions + (direction, )
                score = problem.getCostOfActions(new_actions) + heuristic(coord)
                fringe.push((coord, new_actions), score)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
gs = greedySearch
