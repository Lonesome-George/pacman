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
                    return actions + [direction]
                fringe.push((coord, actions + [direction]))
                visited.append(coord)

    return []

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
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), ()), 0)
    explored = []

    while not fringe.isEmpty():
        node, actions = fringe.pop()

        if problem.isGoalState(node):
            return actions

        explored.append(node)

        for coord, direction, steps in problem.getSuccessors(node):
            if not coord in explored:
                new_actions = actions + (direction, )
                fringe.push((coord, new_actions), problem.getCostOfActions(new_actions))

    return []

    # pQueue = util.PriorityQueue()
    #
    # item = (problem.getStartState(), problem.getStartState(), 0, None)
    # priority = 0
    # pQueue.push( item, priority )
    #
    # visited = set()
    # parentTable = []
    # path = []
    #
    # while (not pQueue.isEmpty()):
    #     state = pQueue.pop()
    #     parentTable.append([state[0], state[1], state[3]])
    #     if (problem.isGoalState(state[1])):
    #         break
    #     currentPoint = state[1]
    #     if (currentPoint not in visited):
    #         visited.add(currentPoint)
    #         for successor in problem.getSuccessors(currentPoint):
    #             cost = state[2] + successor[2]
    #         action = successor[1]
    #         nextPoint = successor[0]
    #         pQueue.push((currentPoint, nextPoint, cost, action), cost)
    #
    # currentState = state[1]
    # while (currentState != problem.getStartState()):
    #     for parent in parentTable:
    #         if (parent[1] == currentState):
    #             currentState = parent[0]
    #             path.insert(0, parent[2])
    #             break
    # return path

  
def nullHeuristic(state):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided searchProblem.  This one is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    closedset = []
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

    return []
  

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
