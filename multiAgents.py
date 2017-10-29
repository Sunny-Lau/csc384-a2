# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        # print legalMoves[chosenIndex]
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        newFoodList = newFood.asList()
        currentFoodList = currentGameState.getFood().asList()

        distanceToFood = [manhattanDistance(x, newPos) for x in newFoodList]
        closestFoodDistance = min(distanceToFood) if distanceToFood else 0

        distanceToGhosts = [manhattanDistance(x.getPosition(), newPos) for x in newGhostStates]
        closestGhostDistance = min(distanceToGhosts) if distanceToGhosts else 0

        score = - closestFoodDistance
        if newPos in currentFoodList:
            score += closestFoodDistance

        if all(t == 0 for t in newScaredTimes):
            if closestGhostDistance < 2:
                score = - 999999

        return score

        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


def manhattanDistance(point1, point2):
    """ return the manattan distance between point1 and point2. """
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def miniMax(state, numMoves):
            player = numMoves % state.getNumAgents()

            bestMove = None
            if state.isWin() or state.isLose() or numMoves == state.getNumAgents() * self.depth:
                return (bestMove, self.evaluationFunction(state))

            if player == 0:
                value = - float("inf")
            else:
                value = float("inf")

            for move in state.getLegalActions(player):
                nxtState = state.generateSuccessor(player, move)
                nxtMove, nxtVal = miniMax(nxtState, numMoves + 1)
                if player == 0 and value < nxtVal:
                    value, bestMove = nxtVal, move
                if player != 0 and value > nxtVal:
                    value, bestMove = nxtVal, move
            return (bestMove, value)

        bestMove, bestVal = miniMax(gameState, 0)

        return bestMove
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(state, alpha, beta, numMoves):
            player = numMoves % state.getNumAgents()

            bestMove = None
            if state.isWin() or state.isLose() or numMoves == state.getNumAgents() * self.depth:
                return (bestMove, self.evaluationFunction(state))

            if player == 0:
                value = - float("inf")
            else:
                value = float("inf")

            for move in state.getLegalActions(player):
                nxtState = state.generateSuccessor(player, move)
                nxtMove, nxtVal = alphaBeta(nxtState, alpha, beta, numMoves + 1)
                if player == 0:
                    if value < nxtVal:
                        value, bestMove = nxtVal, move
                    if value >= beta:
                        return bestMove, value
                    alpha = max(alpha, value)
                if player != 0:
                    if value > nxtVal:
                        value, bestMove = nxtVal, move
                    if value <= alpha:
                        return bestMove, value
                    beta = min(beta, value)
            return bestMove, value

        bestMove, bestVal = alphaBeta(gameState, - float("inf"), float("inf"), 0)
        return bestMove
        # util.raiseNotDefined()
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectiMax(state, numMoves):
            player = numMoves % state.getNumAgents()

            bestMove = None
            if state.isWin() or state.isLose() or numMoves == state.getNumAgents() * self.depth:
                return (bestMove, self.evaluationFunction(state))

            if player == 0:
                value = - float("inf")
            else:
                value = 0

            for move in state.getLegalActions(player):
                nxtState = state.generateSuccessor(player, move)
                nxtMove, nxtVal = expectiMax(nxtState, numMoves + 1)
                if player == 0 and value < nxtVal:
                    value, bestMove = nxtVal, move
                if player != 0:
                    value = value + nxtVal * 1.0 / len(state.getLegalActions(player))
            return (bestMove, value)

        bestMove, bestVal = expectiMax(gameState, 0)

        return bestMove
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPosition = currentGameState.getPacmanPosition()

    foods = currentGameState.getFood().asList()
    foodDistances = [manhattanDistance(f, currentPosition) for f in foods]
    minFood = min(foodDistances) if foodDistances else 0
    avgDistance = sum(foodDistances) / len(foodDistances) if foodDistances else 0

    ghosts = currentGameState.getGhostPositions()
    ghostDistances = [manhattanDistance(g, currentPosition) for g in ghosts]
    minGhost = min(ghostDistances) if ghostDistances else 0

    capsules = currentGameState.getCapsules()
    capsulesDistances = [manhattanDistance(c, currentPosition) for c in capsules]
    minCapsules = min(capsulesDistances) if capsulesDistances else 0

    score = - (1000 * len(foods) + 800 * len(capsules) + 100 * minFood + 10 * avgDistance ) + 10 * minGhost  + 100 * currentGameState.getScore()

    if len(foods) == 1:
        score = - (100 * manhattanDistance(foods[0], currentPosition) + 10 * avgDistance) + 10 * minGhost + 100 * currentGameState.getScore()

    # score += random.randint(1, 80)

    if currentGameState.isWin():
        score = 99999999 
    if currentGameState.isLose():
        score = - 99999999

    return score + random.randint(1, 80)

    # util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction

