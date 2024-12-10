from random import randint
import random
import numpy as np
import collections
import time
import os

def cls2():

    os.system("clear")

class Square(object):
    location = (0, 0)
    isUncovered = False
    value = 0


    def __init__(self, location):
        self.location = location


class MineSweeper(object):
    board = []  
    frontier = []

    row_size = 0
    column_size = 0

    bomb_number = 0

    bomb_value = 9
    covered_value = 10

    num_uncovered = 0
    gameEnd = False

    score = 0
    gameWon = False

    verbose = False

    def __init__(self, row=4, column=4, difficulty=1, verbose=False):
        self.row_size = row
        self.column_size = column

        self.board = []
        self.frontier = []

        self.verbose = verbose

        for row in range(row):
            self.board.append([])
            for col in range(column):
                self.board[row].append(Square((row, col)))         
    
        if difficulty == 1:
            if row * column < 10:
                self.bomb_number = 2
            elif row*column < 20:
                self.bomb_number = 3
            elif row * column < 30:           
                self.bomb_number = 5                        
            elif row * column < 100:
                self.bomb_number = 10           
            else:
                self.bomb_number = 15       
        elif difficulty == 2:
            if row * column < 30:
                self.bomb_number = 10   
            elif row * column < 100:
                self.bomb_number = 15   
            else:
                self.bomb_number = 20
        elif difficulty == 3:   
            if row * column < 30:
                self.bomb_number = 15
            elif row * column < 100:
                self.bomb_number = 20       
            else:
                self.bomb_number = 30
        else:
            raise Exception("Your level input is wrong!")   

        self.insert_mines()

        if verbose:
            print "Playing on %d x %d board with difficulty %d" % \
                (row, column, difficulty)

    def get_state(self):
        state = []
        for row in range(self.row_size):
            for col in range(self.column_size):
                square = self.board[row][col]
                if square.isUncovered == False:
                    state.append(self.covered_value)
                else:
                    state.append(square.value)
        return state


    def rotate_90_clockwise(self, state):
        rotatedState = []
        for i in range(self.row_size):
            for j in range(self.column_size):
                rotatedState.append(state[self.get_state_index_from_location(self.column_size - j - 1, i)])

    def get_state_index_from_location(self, row, col):
        return row*self.row_size + col

    def get_location_from_state_index(self, index):
        return index / self.row_size, index % self.row_size

    def get_label(self):
        label = []
        for row in range(self.row_size):
            for col in range(self.column_size):
                square = self.board[row][col]
                if square in self.frontier:
                    if square.value != self.bomb_value:
                        # could add some more check to determine if square is 
                        # actually a logical choice... + some probability
                        label.append(1)
                    else:
                        label.append(0)
                else:
                    label.append(0)

        return label


    def get_frontier(self):
        return self.frontier


    # uncover all its neighbors.
    def update_board(self, square):
        # Base case: reached a square that was previously uncovered. So, just return.
        if square.isUncovered == True:
            return

        # We are uncovering a square, so if it was in  self.frontier it can no longer be
        if square in self.frontier:
            self.frontier.remove(square)

        # uncover current square
        square.isUncovered = True
        self.num_uncovered += 1

        # Recursive case: uncover all neighbors
        if square.value == 0:
            # Get the neighbors if the square.value is 0
            for neighbor in self.get_neighbors(square).values():
                self.update_board(neighbor)
        # We are not going to uncover this square, so we need to update self.frontier by
        # adding all the neighbors of the current square if not already in self.frontier
        else:
            for neighbor in self.get_neighbors(square).values():
                # Add neighbors to frontier if not already uncovered and not already in frontier
                if neighbor.isUncovered == False and neighbor not in self.frontier:
                    self.frontier.append(neighbor)

    def get_init_state(self):
        state = []
        for i in range(self.row_size*self.column_size):
            state.append(self.covered_value)

        return state

    def get_next_state(self, square):
        if not square.isUncovered:
            if square.value == self.bomb_value:
                self.gameEnd = True
            else:
                self.score += 5
                self.update_board(square)

        # if all non-bomb squares have been uncovered, game is won
        if self.num_uncovered == self.row_size*self.column_size - self.bomb_number:
            self.gameEnd = True
            self.gameWon = True

        if self.verbose:
            print self.get_state()
            print self.get_label()

        return self.get_state()

    def get_neighbors(self, square):
        location = square.location
        neighbors = {}
        i = 0
        for row in range(location[0]-1, location[0]+2):
            for col in range(location[1]-1, location[1]+2):
                if row == location[0] and col == location[1]:
                    continue
                if row >= 0 and row < self.row_size and col >= 0 and col < self.column_size:
                    neighbors[i] = self.board[row][col]
                i = i+1
        #print neighbors
        return neighbors




    #Insert specified number of mines into the area, increase numbers of its neigbours.
    def insert_mines(self):
        bombs = random.sample(range(0, self.row_size*self.column_size), self.bomb_number)
        bomb_positions = [(bomb/self.row_size, bomb % self.row_size) for bomb in bombs]
        
        for bomb in bomb_positions:
            self.board[bomb[0]][bomb[1]].value = self.bomb_value

        for bomb_position in bomb_positions:
            bomb = self.board[bomb_position[0]][bomb_position[1]]
            neigbourlist = self.get_neighbors(bomb)
            
            #increase proper neighbours one
            for neigbour in neigbourlist.values():
                if neigbour.value != self.bomb_value:
                    neigbour.value += 1

    #always choose a corner for the first move. (kind of cheating)
    def first_move(self, corner):
        return {
            0: 0,
            1: self.row_size - 1,
            2: self.row_size*self.column_size - 1 - self.row_size - 1,
            3: self.row_size*self.column_size - 1,
        }.get(corner, 0)

    # User interface: return whether move will lead to a bomb
    def is_bomb(self, square):
        return square.value == self.bomb_value

    def get_square(self, location):
        return self.board[location[0]][location[1]]

    def get_random_move_off_frontier(self):
        randomLocation = (randint(0, self.row_size-1), randint(0, self.column_size-1))
        while randomLocation in self.get_frontier() or self.get_square(randomLocation).isUncovered:
            randomLocation = (randint(0, self.row_size-1), randint(0, self.column_size-1))
        return self.get_square(randomLocation)

    def get_prob_frontier(self):
        frontier = self.get_frontier()
        l=[]
        maxval=999
            # nextMove = random.choice(frontier)
        for sq in frontier:
            # print sq.value
            #find 8 adjacent squares
            p = 0
            curr_loc=sq.location
            x=[0,1,-1]
            next_loc = [0, 0]
            for ii in x:
                for jj in x:
                    s = curr_loc[0] + ii
                    t = curr_loc[1] + jj
                    if s < 0 or t < 0 or s > (self.row_size - 1) or t > (self.column_size - 1):
                        continue
                    else:
                        next_loc[0]=s
                        next_loc[1]=t
                        #print tuple(next_loc)
                        if (self.get_square(tuple(next_loc)).isUncovered==True):
                            v=self.get_square(tuple(next_loc)).value
                            # print v
                            p = p + v
                        # print next_loc
                #print next_loc 
            if p < maxval:
                maxval = p
                # print tuple(next_loc),maxval,p
                nextMove = sq
            # print nextMove.location
        return nextMove


def generate_state_map_by_random_playing(num_total_simulations=100, row=4, col=4, difficulty=1, rewardValue=1):

    qMap = collections.Counter()
    end_time=time.time()+num_total_simulations
    # for i in xrange(num_total_simulations):
    i=0
    start_now = time.strftime("%S", time.localtime(time.time()))
    prev_now=start_now
    timeinsec=0
    print 'Training Time::',num_total_simulations,' seconds\n'
    while time.time()<end_time:
        now = time.strftime("%S", time.localtime(time.time()))
        if(prev_now!=now):
            print 'Running for ',(timeinsec+1),' seconds'
            prev_now=now
            timeinsec+=1
        game = MineSweeper(row, col, difficulty)
        location = (0, 0)
        # location = (randint(0, game.row_size-1), randint(0, game.column_size-1))
        nextMove = game.get_square(location)
        currentState = game.get_init_state()
        #print currentState
        reward = 0
        count_=0
        # print 'New game'
        while True:
            count_+=1
            # print count_
            # print 
            reward = rewardValue if not game.is_bomb(nextMove) else -1*rewardValue
            stateAndAction = (tuple(currentState), nextMove.location)
            #print stateAndAction
            qMap[stateAndAction] += reward
            currentState = game.get_next_state(nextMove)

            if game.gameEnd:
                break

            frontier = game.get_frontier()
            nextMove = random.choice(frontier)
 
        i+=1   
    return qMap,i


#Supervised
def generate_state_map_using_label(num_total_simulations=100, row=4, col=4, difficulty=1, reward=1):
    qMap = collections.Counter()
    end_time=time.time()+num_total_simulations
    # for i in xrange(num_total_simulations):
    i=0
    while time.time()<end_time:
    # for iterationNo in xrange(num_total_simulations):
        # if iterationNo % 1000 == 0:
        #     print "Playing %dth training game." % iterationNo

        game = MineSweeper(row, col, difficulty)

        topLeftCorner = (0, 0)
        nextMove = game.get_square(topLeftCorner)
        currentState = tuple(game.get_next_state(nextMove))

        while not game.gameEnd:
            label = game.get_label()    # get list of correct moves
            listOfCorrectMoveIndices = []
            qMap[currentState] += 1     # to indicate that the algorithm has visited this state once 
            for j in range(len(label)):
                stateAndAction = (currentState, game.get_location_from_state_index(j))
                if label[j] == 1:
                    qMap[stateAndAction] += reward
                    listOfCorrectMoveIndices.append(j)

            nextMove = None

            if not listOfCorrectMoveIndices:
                nextMove = game.get_random_move_off_frontier()
            else:
                index = random.choice(listOfCorrectMoveIndices)
                randomCorrectLocation = game.get_location_from_state_index(index)
                nextMove = game.get_square(randomCorrectLocation)
            
            currentState = tuple(game.get_next_state(nextMove))
        i+=1        
    return qMap,i

# Returns the square of the best move
def getNextMove(qMap, game):
    bestMoveLocation = (-1, -1)
    maxQValue = float("-inf")
    
    possibleMoves = map(lambda x: x.location, game.get_frontier())
    currentState = tuple(game.get_state())
    shouldPickRandomMove = True
    
    for move in possibleMoves:
        q = qMap[ (currentState, move) ]
        if q > maxQValue:
            bestMoveLocation, maxQValue = move, q

    if maxQValue > 0 or qMap[currentState] == 0:
        shouldPickRandomMove = False

    if shouldPickRandomMove:
        return game.get_prob_frontier()
        # return game.get_random_move_off_frontier()

    else:
        return game.get_square(bestMoveLocation)
