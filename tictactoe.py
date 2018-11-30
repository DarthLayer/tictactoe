import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import pprint
import random

def TicTacToeGame(object):
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
        for j in range(self.board.shape[1])