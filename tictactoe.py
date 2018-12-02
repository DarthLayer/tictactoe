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

    def __set_model(self):
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
    
    def __move_selector(model,current_board_state,turn_monitor):
        """
        Function that selects the next move to make from a set of possible legal moves
        Args:
        model: The Evaluator function to use to evaluate each possible next board state
        turn_monitor: 1 if it's the player who places the mark 1's turn to play, 0 if its his opponent's turn
        Returns:
        selected_move: The numpy array coordinates where the player should place thier mark
        new_board_state: The flattened new board state resulting from performing above selected move
        score: The score that was assigned to the above selected_move by the Evaluator (model)

        """
        tracker = {}
        legal_moves_dict = legal_moves_generator(current_board_state,turn_monitor)
        for legal_move_coord in legal_moves_dict:
            score = self.model.predict(legal_moves_dict[legal_move_coord].reshape(1,9))
            tracker[legal_move_coord] = score
        selected_move = max(tracker,key=tracker.get)
        new_board_state = legal_moves_dict[selected_move]
        score = tracker[selected_move]
        return selected_move, new_board_state, score
    
    def row_winning_move_check(current_board_state,legal_moves_dict,turn_monitor):
    """Function to scan rowwise and identify coordinate amongst the legal coordinates that will
    result in a winning board state

    Args:
    legal_moves_dict: Dictionary of legal next moves
    turn_monitor: whose turn it is to move
    
    Returns:
    selected_move: The coordinates of numpy array where placing the 0 will lead to win for the opponent
    """ 
    legal_move_coords =  list(legal_moves_dict.keys())
    random.shuffle(legal_move_coords)
    for legal_move_coord in legal_move_coords:
        current_board_state_copy=current_board_state.copy()
        current_board_state_copy[legal_move_coord]=turn_monitor
        #check for a win along rows
        for i in range(current_board_state_copy.shape[0]):
            if 2 not in current_board_state_copy[i,:] and len(set(current_board_state_copy[i,:]))==1:
                selected_move=legal_move_coord
                return selected_move
            
    def column_winning_move_check(current_board_state,legal_moves_dict,turn_monitor):
        """Function to scan column wise and identify coordinate amongst the legal coordinates that will
        result in a winning board state

        Args:
        legal_moves_dict: Dictionary of legal next moves
        turn_monitor: whose turn it is to move

        Returns:
        selected_move: The coordinates of numpy array where placing the 0 will lead to win for the opponent
        """ 
        legal_move_coords =  list(legal_moves_dict.keys())
        random.shuffle(legal_move_coords)
        for legal_move_coord in legal_move_coords:
            current_board_state_copy=current_board_state.copy()
            current_board_state_copy[legal_move_coord]=turn_monitor
            for j in range(current_board_state_copy.shape[1]):
                        if 2 not in current_board_state_copy[:,j] and len(set(current_board_state_copy[:,j]))==1:
                            selected_move=legal_move_coord
                            return selected_move

    def diag1_winning_move_check(current_board_state,legal_moves_dict,turn_monitor):
        """Function to scan diagonal and identify coordinate amongst the legal coordinates that will
        result in a winning board state

        Args:
        legal_moves_dict: Dictionary of legal next moves
        turn_monitor: whose turn it is to move

        Returns:
        selected_move: The coordinates of numpy array where placing the 0 will lead to win for the opponent

        """ 
        legal_move_coords =  list(legal_moves_dict.keys())
        random.shuffle(legal_move_coords)
        for legal_move_coord in legal_move_coords:
            current_board_state_copy=current_board_state.copy()
            current_board_state_copy[legal_move_coord]=turn_monitor
            if 2 not in np.diag(current_board_state_copy) and len(set(np.diag(current_board_state_copy)))==1:
                selected_move=legal_move_coord
                return selected_move
                
    def diag2_winning_move_check(current_board_state,legal_moves_dict,turn_monitor):
        """Function to scan second diagonal and identify coordinate amongst the legal coordinates that will
        result in a winning board state

        Args:
        legal_moves_dict: Dictionary of legal next moves
        turn_monitor: whose turn it is to move

        Returns:
        selected_move: The coordinates of numpy array where placing the 0 will lead to win for the opponent

        """ 
        legal_move_coords =  list(legal_moves_dict.keys())
        random.shuffle(legal_move_coords)
        for legal_move_coord in legal_move_coords:
            current_board_state_copy=current_board_state.copy()
            current_board_state_copy[legal_move_coord]=turn_monitor
            if 2 not in np.diag(np.fliplr(current_board_state_copy)) and len(set(np.diag(np.fliplr(current_board_state_copy))))==1:
                selected_move=legal_move_coord
                return selected_move
                
    #------------#

    def row_block_move_check(current_board_state,legal_moves_dict,turn_monitor):
        """Function to scan rowwise and identify coordinate amongst the legal coordinates 
        that will prevent the program 
        from winning

        Args:
        legal_moves_dict: Dictionary of legal next moves
        turn_monitor: whose turn it is to move
        
        Returns:
        selected_move: The coordinates of numpy array where placing the 0 will block 1 from winning

        """ 
        legal_move_coords =  list(legal_moves_dict.keys())
        random.shuffle(legal_move_coords)
        for legal_move_coord in legal_move_coords:
            current_board_state_copy=current_board_state.copy()
            current_board_state_copy[legal_move_coord]=turn_monitor
            for i in range(current_board_state_copy.shape[0]):
                if 2 not in current_board_state_copy[i,:] and (current_board_state_copy[i,:]==1).sum()==2:
                    if not (2 not in current_board_state[i,:] and (current_board_state[i,:]==1).sum()==2):
                        selected_move=legal_move_coord
                        return selected_move
                
    def column_block_move_check(current_board_state,legal_moves_dict,turn_monitor):
        """Function to scan column wise and identify coordinate amongst the legal coordinates that will prevent 1 
        from winning

        Args:
        legal_moves_dict: Dictionary of legal next moves
        turn_monitor: whose turn it is to move
        
        Returns:
        selected_move: The coordinates of numpy array where placing the 0 will block 1 from winning

        """ 
        legal_move_coords =  list(legal_moves_dict.keys())
        random.shuffle(legal_move_coords)
        for legal_move_coord in legal_move_coords:
            current_board_state_copy=current_board_state.copy()
            current_board_state_copy[legal_move_coord]=turn_monitor
            
            for j in range(current_board_state_copy.shape[1]):
                        if 2 not in current_board_state_copy[:,j] and (current_board_state_copy[:,j]==1).sum()==2:
                            if not (2 not in current_board_state[:,j] and (current_board_state[:,j]==1).sum()==2):
                                selected_move=legal_move_coord
                                return selected_move

    def diag1_block_move_check(current_board_state,legal_moves_dict,turn_monitor):
        """Function to scan diagonal 1 and identify coordinate amongst the legal coordinates that will prevent 1 
        from winning

        Args:
        legal_moves_dict: Dictionary of legal next moves
        turn_monitor: whose turn it is to move
        
        Returns:
        selected_move: The coordinates of numpy array where placing the 0 will block 1 from winning

        """ 
        legal_move_coords =  list(legal_moves_dict.keys())
        random.shuffle(legal_move_coords)
        for legal_move_coord in legal_move_coords:
            current_board_state_copy=current_board_state.copy()
            current_board_state_copy[legal_move_coord]=turn_monitor    
            if 2 not in np.diag(current_board_state_copy) and (np.diag(current_board_state_copy)==1).sum()==2:
                    if not (2 not in np.diag(current_board_state) and (np.diag(current_board_state)==1).sum()==2):
                        selected_move=legal_move_coord
                        return selected_move
                
    def diag2_block_move_check(current_board_state,legal_moves_dict,turn_monitor):
        """Function to scan second diagonal wise and identify coordinate amongst the legal coordinates that will
        result in a column having only 0s

        Args:
        legal_moves_dict: Dictionary of legal next moves
        turn_monitor: whose turn it is to move
        
        Returns:
        selected_move: The coordinates of numpy array where placing the 0 will lead to two 0s being there (and no 1s)

        """ 
        legal_move_coords =  list(legal_moves_dict.keys())
        random.shuffle(legal_move_coords)
        for legal_move_coord in legal_move_coords:
            current_board_state_copy=current_board_state.copy()
            current_board_state_copy[legal_move_coord]=turn_monitor
            if 2 not in np.diag(np.fliplr(current_board_state_copy)) and (np.diag(np.fliplr(current_board_state_copy))==1).sum()==2:
                if not (2 not in np.diag(np.fliplr(current_board_state)) and (np.diag(np.fliplr(current_board_state))==1).sum()==2):
                    selected_move=legal_move_coord
                    return selected_move

    #---------------#
    def row_second_move_check(current_board_state,legal_moves_dict,turn_monitor):
        """Function to scan rowwise and identify coordinate amongst the legal coordinates that will
        result in a row having two 0s and no 1s

        Args:
        legal_moves_dict: Dictionary of legal next moves
        turn_monitor: whose turn it is to move
        
        Returns:
        selected_move: The coordinates of numpy array where placing the 0 will lead to two 0s being there (and no 1s)

        """ 
        legal_move_coords =  list(legal_moves_dict.keys())
        random.shuffle(legal_move_coords)
        for legal_move_coord in legal_move_coords:
            current_board_state_copy=current_board_state.copy()
            current_board_state_copy[legal_move_coord]=turn_monitor
            
            for i in range(current_board_state_copy.shape[0]):
                if 1 not in current_board_state_copy[i,:] and (current_board_state_copy[i,:]==0).sum()==2:
                    if not (1 not in current_board_state[i,:] and (current_board_state[i,:]==0).sum()==2):
                        selected_move=legal_move_coord
                        return selected_move
                
    def column_second_move_check(current_board_state,legal_moves_dict,turn_monitor):
        """Function to scan column wise and identify coordinate amongst the legal coordinates that will
        result in a column having two 0s and no 1s

        Args:
        legal_moves_dict: Dictionary of legal next moves
        turn_monitor: whose turn it is to move
        
        Returns:
        selected_move: The coordinates of numpy array where placing the 0 will lead to two 0s being there (and no 1s)

        """ 
        legal_move_coords =  list(legal_moves_dict.keys())
        random.shuffle(legal_move_coords)
        for legal_move_coord in legal_move_coords:
            current_board_state_copy=current_board_state.copy()
            current_board_state_copy[legal_move_coord]=turn_monitor
            
            for j in range(current_board_state_copy.shape[1]):
                        if 1 not in current_board_state_copy[:,j] and (current_board_state_copy[:,j]==0).sum()==2:
                            if not (1 not in current_board_state[:,j] and (current_board_state[:,j]==0).sum()==2):
                                selected_move=legal_move_coord
                                return selected_move

    def diag1_second_move_check(current_board_state,legal_moves_dict,turn_monitor):
        """Function to scan diagonal wise and identify coordinate amongst the legal coordinates that will
        result in a column having two 0s and no 1s

        Args:
        legal_moves_dict: Dictionary of legal next moves
        turn_monitor: whose turn it is to move
        
        Returns:
        selected_move: The coordinates of numpy array where placing the 0 will lead to two 0s being there (and no 1s)

        """ 
        legal_move_coords =  list(legal_moves_dict.keys())
        random.shuffle(legal_move_coords)
        for legal_move_coord in legal_move_coords:
            current_board_state_copy=current_board_state.copy()
            current_board_state_copy[legal_move_coord]=turn_monitor
            if 1 not in np.diag(current_board_state_copy) and (np.diag(current_board_state_copy)==0).sum()==2:
                if not (1 not in np.diag(current_board_state) and (np.diag(current_board_state)==0).sum()==2):
                    selected_move=legal_move_coord
                    return selected_move
                
    def diag2_second_move_check(current_board_state,legal_moves_dict,turn_monitor):
        """Function to scan second diagonal wise and identify coordinate amongst 
        the legal coordinates that will result in a column having two 0s and no 1s

        Args:
        legal_moves_dict: Dictionary of legal next moves
        turn_monitor: whose turn it is to move
        
        Returns:
        selected_move: The coordinates of numpy array where opponent places their mark

        """ 
        legal_move_coords =  list(legal_moves_dict.keys())
        random.shuffle(legal_move_coords)
        for legal_move_coord in legal_move_coords:
            current_board_state_copy=current_board_state.copy()
            current_board_state_copy[legal_move_coord]=turn_monitor
            if 1 not in np.diag(np.fliplr(current_board_state_copy)) and (np.diag(np.fliplr(current_board_state_copy))==0).sum()==2:
                if not (1 not in np.diag(np.fliplr(current_board_state)) and (np.diag(np.fliplr(current_board_state))==0).sum()==2):
                    selected_move=legal_move_coord
                    return selected_move
        
    def opponent_move_selector(current_board_state,turn_monitor,mode):
        """Function that picks a legal move for the opponent

        Args:
        current_board_state: Current board state
        turn_monitor: whose turn it is to move
        mode: whether hard or easy mode

        Returns:
        selected_move: The coordinates of numpy array where placing the 0 will lead to two 0s being there (and no 1s)

        """ 
        legal_moves_dict=legal_moves_generator(current_board_state,turn_monitor)
        
        winning_move_checks=[row_winning_move_check,column_winning_move_check,diag1_winning_move_check,diag2_winning_move_check]
        block_move_checks=[row_block_move_check,column_block_move_check,diag1_block_move_check,diag2_block_move_check]
        second_move_checks=[row_second_move_check,column_second_move_check,diag1_second_move_check,diag2_second_move_check]

        if mode=="Hard":
            random.shuffle(winning_move_checks)
            random.shuffle(block_move_checks)
            random.shuffle(second_move_checks)        
            
            for fn in winning_move_checks:
                if fn(current_board_state,legal_moves_dict,turn_monitor):
                    return fn(current_board_state,legal_moves_dict,turn_monitor)
                
            for fn in block_move_checks:
                if fn(current_board_state,legal_moves_dict,turn_monitor):
                    return fn(current_board_state,legal_moves_dict,turn_monitor)
                
            for fn in second_move_checks:
                if fn(current_board_state,legal_moves_dict,turn_monitor):
                    return fn(current_board_state,legal_moves_dict,turn_monitor)
                
            selected_move=random.choice(list(legal_moves_dict.keys()))
            return selected_move
        
        elif mode=="Easy":
            legal_moves_dict=legal_moves_generator(current_board_state,turn_monitor)
            selected_move=random.choice(list(legal_moves_dict.keys()))
            return selected_move
    
    def train(model,mode,print_progress=False):
        """Function trains the Evaluator (model) by playing a game against an opponent 
        playing random moves, and updates the weights of the model after the game
        
        Note that the model weights are updated using SGD with a batch size of 1

        Args:
        model: The Evaluator function being trained

        Returns:
        model: The model updated using SGD
        y: The corrected scores

        """ 
        # start the game
        if print_progress == True:
            print("___________________________________________________________________")
            print("Starting a new game")
        game = TicTacToeGame()
        game.toss()
        scores_list = []
        corrected_scores_list = []
        new_board_states_list = []
        
        while(1):
            if game.game_status() == "In Progress" and game.turn_monitor == 1:
                # If its the program's turn, use the Move Selector function to select the next move
                selected_move,new_board_state,score = game.__move_selector(model,game.board,game.turn_monitor)
                scores_list.append(score[0][0])
                new_board_states_list.append(new_board_state)
                # Make the next move
                game.game_status,game.board = game.move(game.turn_monitor,selected_move)
                if print_progress == True:
                    print("Program's Move")
                    print(board)
                    print("\n")
            elif game.game_status() == "In Progress" and game.turn_monitor == 0:
                selected_move = game.opponent_move_selector(game.board,game.turn_monitor,mode=mode)
            
                # Make the next move
                game.game_status,game.board=game.move(game.turn_monitor,selected_move)
                if print_progress==True:
                    print("Opponent's Move")
                    print(board)
                    print("\n")
            else:
                break

        
        # Correct the scores, assigning 1/0/-1 to the winning/drawn/losing final board state, 
        # and assigning the other previous board states the score of their next board state
        new_board_states_list = tuple(new_board_states_list)
        new_board_states_list = np.vstack(new_board_states_list)
        if game.game_status == "Won" and (1-game.turn_monitor)==1: 
            corrected_scores_list = shift(scores_list,-1,cval=1.0)
            result="Won"
        if game.game_status=="Won" and (1-game.turn_monitor)!=1:
            corrected_scores_list=shift(scores_list,-1,cval=-1.0)
            result="Lost"
        if game.game_status=="Drawn":
            corrected_scores_list=shift(scores_list,-1,cval=0.0)
            result="Drawn"
        if print_progress==True:
            print("Program has ",result)
            print("\n Correcting the Scores and Updating the model weights:")
            print("___________________________________________________________________\n")
            
        x=new_board_states_list
        y=corrected_scores_list
        
        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]
        
        # shuffle x and y in unison
        x,y=unison_shuffled_copies(x,y)
        x=x.reshape(-1,9) 
        
        # update the weights of the model, one record at a time
        game.model.fit(x,y,epochs=1,batch_size=1,verbose=0)
        return model,y,result
    
    
        game_counter=1
        data_for_graph=pd.DataFrame()

        mode_list=["Easy","Hard"]

        while(game_counter<=250000):
            mode_selected=np.random.choice(mode_list, 1, p=[0.5,0.5])
            model,y,result=train(model,mode=mode_selected[0],print_progress=False)
            data_for_graph=data_for_graph.append({"game_counter":game_counter,"result":result},ignore_index=True)
            if game_counter % 10000 == 0:
                print("game#:",game_counter)
                print(mode_selected[0])
            game_counter+=1