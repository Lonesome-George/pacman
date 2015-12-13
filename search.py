#coding=utf-8

# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    movements = dfsMain(problem, start)
    return movements

def dfsMain(problem, start):
    opened = []
    closed = []
    states = {} # state dict
    goal = None # goal state
    opened.append(start)
    while True:
        current = opened.pop()
        closed.append(current)
        if problem.isGoalState(current):
            goal = current
            break
        else:
            succesors = problem.getSuccessors(current)
            for succesor in succesors:
                if succesor[0] not in closed:
                    opened.append(succesor[0])
                    states[succesor[0]] = current # record state
    movements = [] # construct reversed movements
    current = goal
    while True:
        parent = states[current]
        movements.append(getDirection(parent, current))
        current = parent
        if current == start: break
    return list(reversed(movements))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    movements = bfsMain(problem, start)
    return movements

def bfsMain(problem, start):
    opened = []
    closed = []
    states = {} # state dict
    goal = None # goal state
    opened.append(start)
    while len(opened) > 0:
        current = opened.pop(0)
        closed.append(current)
        if problem.isGoalState(current):
            goal = current
            break
        else:
            succesors = problem.getSuccessors(current)
            for succesor in succesors:
                if succesor[0] not in closed:# or succesor[0] not in opened:
                    opened.append(succesor[0])
                    states[succesor[0]] = current # record state
    movements = [] # construct reversed movements
    current = goal
    while True:
        parent = states[current]
        movements.append(getDirection(parent, current))
        current = parent
        if current == start: break
    return list(reversed(movements))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    movements = ucsMain(problem, start)
    return movements

def ucsMain(problem, start):
    opened = [] # format [(state, cost),...]
    closed = [] # format [state,...]
    states = {} # states {state: ([path, cost]),...}
    goal = None # goal state
    opened.append((start, 0))
    states[start] = ([start], 0) # start to start
    while len(opened) > 0:
        current = find_smallest_cost_node(opened)
        opened.remove(current)
        if problem.isGoalState(current[0]):
            goal = current[0]
        closed.append(current[0])
        succesors = problem.getSuccessors(current[0])
        for succesor in succesors:
            if succesor[0] not in closed:# or succesor[0] not in opened:
                step_cost = succesor[2]
                path_cost = states[current[0]][1] + step_cost
                opened.append((succesor[0], path_cost))
                path = states[current[0]][0][:] # copy list
                path.append(succesor[0])
                if not states.has_key(succesor[0]) or path_cost < states[succesor[0]][1]:
                    states[succesor[0]] = (path, path_cost)
    movements = [] # construct movements
    path = states[goal][0]
    for i in range(len(path) - 1):
        cur = path[i]
        next = path[i+1]
        movements.append(getDirection(cur, next))
    return movements

def find_smallest_cost_node(nodes):
    idx = 0
    min_cost = nodes[0][1]
    for i in range(0, len(nodes)):
        if nodes[i][1] < min_cost:
            idx = i
            min_cost = nodes[i][1]
    return nodes[idx]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    movements = aStarMain(problem, heuristic)
    return movements

def aStarMain(problem, heuristic):
    start = problem.getStartState()
    opened = [] # format [(state, heur_cost),...]
    closed = [] # format [state,...]
    states = {} # states {state: ([path, cost]),...}
    goal = None # goal state
    opened.append((start, heuristic(start, problem)))
    states[start] = ([start], 0) # start to start
    while len(opened) > 0:
        current = find_smallest_cost_node(opened)
        opened.remove(current)
        if problem.isGoalState(current[0]):
            goal = current[0]
            # break
        closed.append(current[0])
        succesors = problem.getSuccessors(current[0])
        for succesor in succesors:
            if succesor[0] not in closed:# or succesor[0] not in opened:
                path_cost = states[current[0]][1] + succesor[2]
                heur_dist = heuristic(current[0], problem) # 启发式距离
                opened.append((succesor[0], path_cost + heur_dist))
                path = states[current[0]][0][:] # copy list
                path.append(succesor[0])
                if not states.has_key(succesor[0]) or path_cost < states[succesor[0]][1]:
                    states[succesor[0]] = (path, path_cost)
    movements = [] # construct movements
    path = states[goal][0]
    for i in range(len(path) - 1):
        cur = path[i]
        next = path[i+1]
        movements.append(getDirection(cur, next))
    return movements

def getDirection(cur, next):
     from game import Directions
     if cur[0] == next[0]:
         if cur[1] < next[1]:
             return Directions.NORTH
         else:
             return Directions.SOUTH
     else:
         if cur[0] < next[0]:
             return Directions.EAST
         else:
             return Directions.WEST


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
