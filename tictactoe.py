import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import pprint
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
#from keras.utils.vis_utils import plot_model
from keras.layers import Dropout

class TicTacToeGame(object):
    def __init__(self):
        self.board = np.full((3,3),2)
    
    def toss(self):
        """
        Function to simulate a toss and decide which player goes first

        Args: N/A

        Returns:
        Returns 1 if players assigned 1 has won, or 0 if opponent assigned 0 has won
        """
        turn = np.random.randint(0,2,size=1)
        if turn == 0:
            self.turn_monitor = 0
        elif turn == 1:
            self.turn_monitor = 1
    
    def move(self,player,coord):
        """
        Function to perform the action of placing a mark on the tictactoe board
        After performing the action, this function flips the value of the turn_monitor
        to the next player
        Args:
            player: 1 if player 1, 0 if player 0
            coord:the coordinate where the 1 or 0 is to be placed
            on the tictactoe board (numpy array)
        Returns:
            game_status(): class the game status function and returns its value
            board: returns the new board state after the move
        """
        if self.board[coord]!=2 or self.game_status()!="In progress" or self.turn_monitor!=player:
            raise ValueError("Invalid move")
        self.board[coord]=player
        self.turn_monitor=1-player #switches to the other player
        return self.game_status(),self.board

    def game_status(self):
        """
        Function to check the current status of the game, whether
        the game has been won, drawn or is in progress
        Args:N/A
        Returns:
        "Won", "Drawn" or "In progress"
        """
        #check for a win along rows
        for i in range(self.board.shape[0]):
             if 2 not in self.board[i,:] and len(set(self.board[i,:]))==1:
                 return "Won"
        #check for a win along columns
        for j in range(self.board.shape[1]):
            if 2 not in self.board[:,j] and len(set(self.board[:,j]))==1:
                return "Won"
        
        #check for a win along diagonals
        if 2 not in np.diag(self.board) and len(set(np.diag(self.board)))==1:
            return "Won"
        if 2 not in np.diag(np.fliplr(self.board)) and len(set(np.diag(np.fliplr(self.board))))==1:
            return "Won"
        
        #check for a draw
        if not 2 in self.board:
            return "Drawn"
        else:
            return "In progress"
        
    def legal_moves_generator(current_board_state,turn_monitor):
        """
        Function that returns the set of all possible legal moves and resulting
        board states, for a given input board state and player
        Args:
            current_board_state: the current board state
            turn_monitor: 1 if its player 1 (places Xs), 0 if its player 0 (places Os)
        Returns:
            legal_moves_dict: a dictionary of a list of possible next coordinate-resulting
            board state pairs
            The resulting board state is flattened to 1d array
        """
        legal_moves_dict = {}
        for i in range(current_board_state.shape[0]):
            for j in range(current_board_state.shape[1]):
                if current_board_state[i,j]==2:
                    board_state_copy = current_board_state.copy()
                    board_state_copy[i,j] = turn_monitor
                    legal_moves_dict[(i,j)] = board_state_copy.flatten()
        return legal_moves_dict

    def set_model(self):
        self.model = Sequential()
        self.model.add(Dense(18,input_dim=9,kernel_initializer='normal',activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(9, kernel_initializer='normal',activation='relu'))
        self.model.add(Dropout(0.1))
        #self.model.add(Dense(9, kernel_initializer='normal',activation='relu'))
        #self.model.add(Dropout(0.1))
        #self.model.add(Dropout(0.1))
        #self.model.add(Dense(5, kernel_initializer='normal',activation='relu'))
        self.model.add(Dense(1,kernel_initializer='normal'))
        self.learning_rate = 0.001
        self.momentum = 0.8
        self.sgd = SGD(lr=self.learning_rate,momentum=self.momentum,nesterov=False)
        self.model.compile(optimizer=self.sgd,loss='mean_squared_error')
        self.model.summary()