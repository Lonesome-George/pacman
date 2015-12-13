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
    fringe = util.PriorityQueue()
    fringe.push( (problem.getStartState(), []), 0)
    explored = []

    while not fringe.isEmpty():
        node, actions = fringe.pop()

        if problem.isGoalState(node):
            return actions

        explored.append(node)

        for coord, direction, steps in problem.getSuccessors(node):
            if not coord in explored:
                new_actions = actions + [direction]
                fringe.push((coord, new_actions), problem.getCostOfActions(new_actions))

    return []

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
    closedset = []
    fringe = util.PriorityQueue()
    start = problem.getStartState()
    fringe.push((start, []), heuristic(start, problem))

    while not fringe.isEmpty():
        node, actions = fringe.pop()

        if problem.isGoalState(node):
            return actions

        closedset.append(node)

        for coord, direction, cost in problem.getSuccessors(node):
            if not coord in closedset:
                new_actions = actions + [direction]
                score = problem.getCostOfActions(new_actions) + heuristic(coord, problem)
                fringe.push((coord, new_actions), score)

    return []

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
