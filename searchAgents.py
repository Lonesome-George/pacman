#coding=utf-8

"""
This file contains all of the agents that can be selected to 
control Pacman.  To select an agent, simple use the '-p' option
when running pacman.py.  That is, to load your DepthFirstSearchAgent,
just run the following command from the command line:

> python pacman.py -p DepthFirstSearchAgent

Please only change the parts of the file where you are asked; look
for the lines that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the
project description for details.

Good luck and happy searching!
"""
from game import Directions
from game import Agent
from game import Actions
from util import manhattanDistance
import util
import time
import search

class GoWestAgent(Agent):
  """
  An agent that goes West until it can't.
  """
  def getAction(self, state):
    """
      The agent receives a GameState (pacman.py).
    """
    if Directions.WEST in state.getLegalPacmanActions():
      return Directions.WEST
    else:
      return Directions.STOP
    
#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py and      #
#           complete the SearchAgent class            #
#######################################################

class PositionSearchProblem(search.SearchProblem):
  """
  A search problem defines the state space, start state, goal test,
  successor function and cost function.  This search problem can be 
  used to find paths to a particular point on the pacman board.
  
  The state space consists of (x,y) positions in a pacman game.
  
  This search problem is fully specified and should not require change.
  """
  
  def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1)):
    """
    Stores the start and goal.  
    
    gameState: A GameState object (pacman.py)
    costFn: A function from a search state (tuple) to a non-negative number
    goal: A position in the gameState
    """
    self.walls = gameState.getWalls()
    self.startState = gameState.getPacmanPosition()
    self.goal = goal
    self.costFn = costFn
    if gameState.getNumFood() != 1 or not gameState.hasFood(*goal):
      print 'Warning: this does not look like a regular search maze'

    # For display purposes
    self._visited, self._visitedlist, self._expanded = {}, [], 0

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
     isGoal = state == self.goal 
     
     # For display purposes only
     if isGoal:
       print 'Goal found after expanding %d nodes.' % self._expanded
       self._visitedlist.append(state)
       import __main__
       if '_display' in dir(__main__):
         if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
           __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable
       
     return isGoal   
   
  def getSuccessors(self, state):
    """
    Returns successor states, the actions they require, and a cost of 1.
    
     As noted in search.py:
         For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
    """
    
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append( ( nextState, action, cost) )
        
    # Bookkeeping for display purposes
    self._expanded += 1 
    if state not in self._visited:
      self._visited[state] = True
      self._visitedlist.append(state)
      
    return successors

  def getCostOfActions(self, actions):
    """Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999"""
    if not actions:
      return 999999
    x,y= self.getStartState()
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]:
        return 999999
      cost += self.costFn((x,y))
    return cost
    
  
class SearchAgent(Agent):
  """
  This very general search agent finds a path using a supplied search algorithm for a
  supplied search problem, then returns actions to follow that path.
  
  As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)
  """
  
  INDEX = 0
  def __init__(self, searchFunction = None, searchType=PositionSearchProblem):
    self.searchFunction = searchFunction
    self.searchType = searchType
    
  def registerInitialState(self, state):
    
    """
    This is the first time that the agent sees the layout of the game board. Here, we
    choose a path to the goal.  In this phase, the agent should compute the path to the
    goal and store it in a local variable.
    
    state: a GameState object (pacman.py)
    """
    
    
    "*** YOUR CODE HERE ***"
    if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
    starttime = time.time()
    problem = self.searchType(state) # Makes a new search problem
    self.actions  = self.searchFunction(problem) # Find a path
    totalCost = problem.getCostOfActions(self.actions)
    print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
    if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)
    # print 'Path found with total cost of %d in %.1f seconds' % (problem.getCostOfActions(self.actions), time.time() - starttime)
    
  def getAction(self, state):
    """
    Returns the next action in the path chosen earlier (in registerInitialState).  Return
    Directions.STOP if there is no further action to take.
    
    state: a GameState object (pacman.py)
    """
    
    "*** YOUR CODE HERE ***"
    if 'actionIndex' not in dir(self): self.actionIndex = 0
    i = self.actionIndex
    self.actionIndex += 1
    if i < len(self.actions):
        return self.actions[i]
    else:
        return Directions.STOP


class TinyMazeSearchAgent(SearchAgent):
  """
  An agent which uses tinyClassSearch to find a path (which only returns a
  correct path for tinyMaze) and follows that path.
  """
  def __init__(self):
    SearchAgent.__init__(self, search.tinyMazeSearch)  


class DepthFirstSearchAgent(SearchAgent):
  """
  An agent that first computes a path to the goal using DFS, then
  follows that path.
  """
  def __init__(self):
    SearchAgent.__init__(self, search.depthFirstSearch)  

class BreadthFirstSearchAgent(SearchAgent):
  """
  An agent that first computes a path to the goal using BFS, then
  follows that path.
  """
  def __init__(self):
    SearchAgent.__init__(self, search.breadthFirstSearch)  
    
class UniformCostSearchAgent(SearchAgent):
  """
  An agent that computes a path to the goal position using UCS.
  """
  def __init__(self):
    SearchAgent.__init__(self, search.uniformCostSearch)  

class StayEastSearchAgent(SearchAgent):
  """
  An agent that computes a path to the goal position using UCS, but
  lets its cost function guide it eastward.
  """
  def __init__(self):
    problem = lambda x: PositionSearchProblem(x, stayEastCost)
    SearchAgent.__init__(self, search.uniformCostSearch, problem)  
    
class StayWestSearchAgent(SearchAgent):
  """
  An agent that computes a path to eat all the dots using UCS, but
  lets its cost function guide it westward.

  """
  def __init__(self):
    problem = lambda x: PositionSearchProblem(x, stayWestCost)
    SearchAgent.__init__(self, search.uniformCostSearch, problem)  

def stayEastCost(position):
  """
  Gives a cost for each (x,y) position that guides a search agent eastward
  """
  return .5 ** position[0]  
  
def stayWestCost(position):
  """
  Gives a cost for each (x,y) position that guides a search agent westward
  """
  return 2 ** position[0]  

class FoodSearchProblem:
  """
  A search problem associated with finding the a path that collects all of the 
  food (dots) in a Pacman game.
  
  A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
    pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
    foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food 
  """
  def __init__(self, state):
    self.start = (state.getPacmanPosition(), state.getFood())
    self.walls = state.getWalls()
    self._expanded = 0
      
  def getStartState(self):
    return self.start
  
  def isGoalState(self, state):
    if state[1].count() == 0:
      print 'Goal found after expanding %d nodes.' % self._expanded
      return True
    return False

  def getSuccessors(self, state):
    """    
    Returns successor states, the actions they require, and a cost of 1.
    
     As noted in search.py:
         For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     
    """
    successors = []
    self._expanded += 1
    for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state[0]
      dx, dy = Actions.directionToVector(direction)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextFood = state[1].copy()
        nextFood[nextx][nexty] = False
        successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
    return successors

  def getCostOfActions(self, actions):
    """Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999"""
    x,y= self.getStartState()[0]
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]:
        return 999999
      cost += 1
    return cost
     
class UniformCostFoodSearchAgent(SearchAgent):
  """
  An agent that computes a path to eat all the dots using UCS.
  """
  def __init__(self):
    SearchAgent.__init__(self, search.uniformCostSearch, FoodSearchProblem)  

def manhattanAStar(problem):
  """
  A wrapper for A* that uses the Manhattan distance heuristic.
  """
  return search.aStarSearch(problem, lambda x: manhattanDistance(x, problem.goal))

class ManhattanAStarSearchAgent(SearchAgent):
  """
  An agent that computes a path to the goal position using AStar and
  the Manhattan distance heuristic.
  """
  def __init__(self):
    SearchAgent.__init__(self, manhattanAStar, PositionSearchProblem)

def trivialFoodHeuristic(state, problem):
  """
   A trivial heuristic for the all-food problem,  which returns 0 for goal states and 1 otherwise.
  """
  if(state[1].count() == 0):
    return 0
  else:
    return 1

###########################################################    
# You have to fill in several parts of the following code #
###########################################################    

def getFoodHeuristic(gameState):
  """
  Instead of filling in the foodHeuristic function directly, you can fill in 
  this function which takes a full gameState for Pacman (see pacman.py) and
  returns a heuristic function.  The heuristic function must
    - take a single parameter, a search state
    - return a non-negative number that is the value of the heuristic at that state

  This function is *only* here for students who want to create more complex 
  heuristics that use aspects of the gameState other than the food Grid and
  Pacman's location (such as where the walls are, etc.)
    
  Note: The state that will be passed to your heuristic function is a tuple 
  ( pacmanPosition, foodGrid ) where foodGrid is a Grid (see game.py) of either 
  True or False.
  """
  # If you don't want to implement this method, you can leave this default implementation
  return foodHeuristic

def greedyHeuristic(state):
  return state[1].count()

# def foodHeuristic(state):
def foodHeuristic(state, problem):
  """
  Here, you can write your food heuristic function instead of using getFoodHeuristic.
  This heuristic must be admissible (if your AStarFoodSearchAgent and your 
  UniformCostSearchAgent *ever* find solutions of different length, your heuristic 
  is *not* admissible).  
  
  The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a 
  Grid (see game.py) of either True or False.
  
  Note that this function *does not* have access to the location of walls, capsules,
  ghosts, etc.  If you want to work with this information, you should implement
  getFoodHeuristic instead of this function.
  
  Hint: getFoodHeuristic can return a heuristic that encapsulates data through a 
  function closure (like the manhattanAStar function above).  If you don't know how 
  this works, come to office hours.
  """
  "*** YOUR CODE HERE ***"
  position, foodGrid = state
  foodList = list(foodGrid.asList())
  if len(foodList) == 0: return 0

  # from search import getShortestDist
  # objList = foodList[:]
  # objList.append(position)
  # mazeDist = {}
  # for i in range(len(objList)): mazeDist[i] = {}
  # for i in range(len(objList)):
  #     obj1 = objList[i]
  #     for j in range(i+1, len(objList)):
  #         obj2 = objList[j]
  #         mazeDist[i][j] = mazeDist[j][i] = getShortestDist(obj1, obj2, problem.walls)
  # min_s_cost = 0
  # if len(objList) > 1:
  #     min_s_cost = min_spanning_cost(len(objList), mazeDist)
  # return min_s_cost

  # Optimized
  # from search import getShortestDist
  # objList = foodList[:]
  # mazeDist = {}
  # for i in range(len(objList)): mazeDist[i] = {}
  # for i in range(len(objList)):
  #     obj1 = objList[i]
  #     for j in range(i+1, len(objList)):
  #         obj2 = objList[j]
  #         mazeDist[i][j] = mazeDist[j][i] = getShortestDist(obj1, obj2, problem.walls)
  # min_s_cost = 0
  # if len(objList) > 1:
  #     min_s_cost = min_spanning_cost(len(objList), mazeDist)
  # dist2spanning_tree = None
  # for food in foodList:
  #     dist = getShortestDist(position, food, problem.walls)
  #     if dist2spanning_tree is None or dist < dist2spanning_tree:
  #         dist2spanning_tree = dist
  # return min_s_cost + dist2spanning_tree

  # Optimized again
  from search import getShortestDist
  mazeDist = {}
  for i in range(len(foodList)): mazeDist[i] = {}
  for i in range(len(foodList)):
      obj1 = foodList[i]
      for j in range(i+1, len(foodList)):
          obj2 = foodList[j]
          mazeDist[i][j] = mazeDist[j][i] = getShortestDist(obj1, obj2, problem.walls)
  min_s_cost = 0
  if len(foodList) > 1:
      min_s_cost, fringe_vertexes = min_spanning_cost(len(foodList), mazeDist)
  else:
      fringe_vertexes = [0]
  dist2spanning_tree = None
  for vertex in fringe_vertexes:
      dist = getShortestDist(position, foodList[vertex], problem.walls)
      if dist2spanning_tree is None or dist < dist2spanning_tree:
          dist2spanning_tree = dist
  return min_s_cost + dist2spanning_tree

def min_spanning_cost(num_vertexes, mazeDist):
    min_cost = 0
    vertexes = set() # 最小生成树中的顶点集
    links = {}       # 每个顶点的连接度
    vertexes.add(0) # add the first vertex
    for i in range(num_vertexes):
        links[i] = 0
    links[0] = 1
    while len(vertexes) < num_vertexes:
        # min_cost += find_min_dist(vertexes, mazeDist)
        startv, endv, min_dist = find_min_dist(vertexes, mazeDist)
        links[startv] += 1
        links[endv] += 1
        min_cost += min_dist
    fringe_vertexes = [] # 边缘顶点集(连接度为1)
    for vertex, count in links.iteritems():
        if count == 1:
            fringe_vertexes.append(vertex)
    return min_cost, fringe_vertexes

def find_min_dist(vertexes, mazeDist):
    min_dist = None
    startv = None # 出发顶点
    endv   = None # 到达顶点
    for v1 in vertexes:
        startv = v1
        for v2, dist in mazeDist[v1].iteritems():
            if v2 not in vertexes and (min_dist is None or dist < min_dist):
                min_dist = dist
                endv = v2
    vertexes.add(endv)
    return startv, endv, min_dist

def trivialSearch(problem):
  return search.aStarSearch(problem, trivialFoodHeuristic)

def foodSearch(problem):
  return search.aStarSearch(problem, foodHeuristic)

class AStarFoodSearchAgent(SearchAgent):
  """
  An agent that computes a path to eat all the dots using AStar.
  
  You should use either foodHeuristic or getFoodHeuristic in your code here.
  """
  "*** YOUR CODE HERE ***"
  def __init__(self, searchFunction=None, searchType=FoodSearchProblem):
    __import__(__name__).getFoodHeuristic = lambda gameState: foodHeuristic
    __import__(__name__).foodHeuristic    = foodHeuristic
    SearchAgent.__init__(self, foodSearch, searchType)


def greedySearch(problem):
  return search.greedySearch(problem, greedyHeuristic)
  

class GreedyFoodSearchAgent(SearchAgent):
  """
  An agent that computes a path to eat all the dots using greedy search.
  """
  def __init__(self, searchFunction = None, searchType = FoodSearchProblem):
    __import__(__name__).getFoodHeuristic = lambda gameState: foodHeuristic
    __import__(__name__).foodHeuristic    = foodHeuristic
    SearchAgent.__init__(self, greedySearch, searchType)
    

class TrivialAStarFoodSearchAgent(AStarFoodSearchAgent):
  """
  An AStarFoodSearchAgent that uses the trivial heuristic instead of the one defined by getFoodHeuristic
  """
  def __init__(self, searchFunction=None, searchType=FoodSearchProblem):
    # Redefine getFoodHeuristic to return the trivial one.
    __import__(__name__).getFoodHeuristic = lambda gameState: trivialFoodHeuristic
    __import__(__name__).foodHeuristic    = trivialFoodHeuristic
    AStarFoodSearchAgent.__init__(self, searchFunction, searchType)
