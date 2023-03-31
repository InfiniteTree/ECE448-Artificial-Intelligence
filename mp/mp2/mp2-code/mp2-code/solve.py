# -*- coding: utf-8 -*-
import numpy as np
import collections
from collections import  defaultdict


def getPTypeAndLable(pent):
    '''
    Function getPTypeAndPLable
    Input: A pent
    Output: Two ints: one represents the label number of the pent matrix and 
                    the other represents the the number of squares that some pent is made by 
    '''
    PLabel, PType = 0, 0
    for row in range(pent.shape[0]):
        for col in range(pent.shape[1]):
            if pent[row][col] != 0:
                PType += 1
                if PLabel == 0:
                    PLabel = pent[row][col]
    return PType, PLabel


def getMutations(pent):
    '''
    Function getMutation
    Input: A pent
    Output: A list contain all of the possible mutations by flipping and rotation of a pent
            with the relative coordinates of each squares
    '''
    mutations = set() # use a set to avoid add reduplicated numpy mutation
    mutation = pent
    rotation_num = 4 # rot90° while in a 2-D plane 360°/90° = 4
    flip_num = 2 # Two possible flipping trials with flip 180°

    i=0
    while i<rotation_num:
        j = 0
        while j<flip_num:
            Tran_mutation = getSingleMutation(mutation)
            mutations.add(Tran_mutation) # add first to add the original pent
            mutation = np.flip(mutation,j-1)
            ### print("mutation is",mutation)
            j += 1
        mutation = np.rot90(mutation)
        i += 1
    return list(mutations)


def getSingleMutation(mutation):
    '''
    Function getSingleMutation
    Input: A mutation
    Output: A tuple contain one mutations by flipping or rotation of a pent
            with the relative coordinates of each squares
    '''
    coords = tuple() # use tuple to avoid np error in add
    for x, y in np.ndindex(mutation.shape):
        if mutation[x][y] != 0:
            coords += ((x, y),)

    origin = coords[0] # Define the origin pos of the mutation to calculate the other squares' coordinates

    for coord in coords:
        if coord[1] < origin[1]:
            origin = coord
        elif coord[1] == origin[1] and coord[0] < origin[0]:
            origin = coord

    TranMutation = tuple()
    for coord in coords:
        TranMutation += ((coord[0] - origin[0], coord[1] - origin[1]),)

    return tuple(sorted(TranMutation)) 


def getPentsMap(pents):
    '''
    Function getPentsMap
    Input: Pents
    Output: A dictionary contain the (type, label) as the key,
            and the coordinates of all mutations by all pents as the value
    '''
    pentsMaps = {}
    p_type = 0 # (2/3/5) the number of squares that some pent is made by 
    p_label = 0 # for pento:1-12
    for pent in pents:
        p_type, p_label = getPTypeAndLable(pent)
        pentsMaps[p_label] = getMutations(pent)
    return pentsMaps

def in_bound(board_size, pent):
    '''
    Function: in_bound
    Instruction: Help to judge whether a pent to place is in bound
    Input: board_size, pent
    Output: A bool value: True indicates in bound while false for out of bound
    '''
    return 0 <= pent[0] < board_size[0] and 0 <= pent[1] < board_size[1]

def GDFS_forward_checking(board, cur_y):
    '''
    Function GDFS_forward_checking
    Input: the current board, the current position x, the current position y
    Output: A bool value to indicate whether the forward checking is successful
    '''
    fc_flag = True
    for i in range(cur_y):
        for j in range(board.shape[0]):
            if board[i][j] == -1: # blank point exits previously
                fc_flag = False
                break
    return fc_flag

def pents_Gdfs(board, pents_maps,  p_type, dfs_call):
    '''
    Function: pents_dfs
    Instruction: Using GDFS algorithm doing recursion to find a accessible solution:
                 Consider to fix all the points of the board one by one with the order from left to right
                 row by row. Use Heuristics that 
                 1)choose the next variable(the point) to asign by LRV 
                 2) use early detection of failure by use forward checking.
    Input: board_size, pent
    Output: a tuple with two variables:
            a bool value: True indicates in bound while false for out of bound
            a new board: a board that is resulted with inserted pents on it
    '''
    find_status = False # a flag for indicating whether successful to find a final solution
    next_loop = False # a flag for next loop in travesal one pents_map
    dfs_call[0]+=1
    
    # The pents_maps is empty
    if not pents_maps:
        ### print("dfs time is", dfs_call[0])
        
        find_status = True
        return find_status, board

    # Traversal all of the points in the board matrix to put the pents into it
    for x, y in ((x, y) for y in range(board.shape[1]) for x in range(board.shape[0])):
        # First find the the uncovered position to add a pent
        if board[x][y] == 0:
            # Use each pent with smallest mutations one by one to check whether it can be fit in the board (LRV)
            for plabel, mutations in pents_maps.items():
                for mutation in mutations:
                    next_loop = False
                    pent = [0, 0]
                    # check whether it's accessible to add the pent onto the board
                    for x_move, y_move in mutation:
                        pent[0] = x + x_move
                        pent[1] = y + y_move
                        if not in_bound(board.shape, pent) or board[x + x_move][y + y_move] != 0:
                            next_loop = True
                            break
                        # Use forward checking to do early detection of failure solution
                        if x == 0: # Check in the start of a new row
                            if y == p_type:
                                if not GDFS_forward_checking(board, y):
                                    next_loop = True
                                    break
                    # continue to the next mutation fixing if out of board or fail insertion detected
                    if next_loop:
                        continue

                    # data updated for next recursion
                    new_board = np.array(board)
                    new_pents_maps = dict(pents_maps)
                    new_pents_maps.pop(plabel)
                    for x_move, y_move in mutation:
                        new_board[x + x_move][y + y_move] = plabel # add the pent to the board
                    
                    # dfs Recursion: forward-tracking
                    find_status, res_board = pents_Gdfs(new_board, new_pents_maps,  p_type, dfs_call)

                    if find_status:
                        return find_status, res_board

            return find_status, board


def solve(board, pents):
    """
    This is the function you will implement. It will take in a numpy array of the board
    as well as a list of n tiles in the form of numpy arrays. The solution returned
    is of the form [(p1, (row1, col1))...(pn,  (rown, coln))]
    where pi is a tile (may be rotated or flipped), and (rowi, coli) is 
    the coordinate of the upper left corner of pi in the board (lowest row and column index 
    that the tile covers).
    
    -Use np.flip and np.rot90 to manipulate pentominos.
    
    -You may assume there will always be a solution.
    """
    board = np.array(board)

    # change the board a litte to be compatible with the pents
    board[board == 0] = -1
    board[board == 1] = 0
    
    #print("pents_maps is",pents_map)
    pents_maps = getPentsMap(pents)
    p_type = getPTypeAndLable(pents[0])[0]

    init_call = [0] # Use a list to counter the number of times call by recursive pents_dfs. list:Variable type
    res_status, res_board = pents_Gdfs(board, dict(pents_maps), p_type, init_call)
    ### print(res_status)
    print(res_board)  # UI print for the tiled board
    
    
    if not res_status:
        print("Failed: Could not found a solution")
        return
    
    elif res_status:
        # use the board to determine the final layout to return
        pents_coords = defaultdict(list)
        for x, y in np.ndindex(res_board.shape):
            if res_board[x][y] in pents_maps:
                pents_coords[res_board[x][y]].append((x, y))

        sol = []

        for plabel, coords in pents_coords.items():
            min_x, max_x, min_y, max_y = coords[0][0], coords[0][0], coords[0][1], coords[0][1]

            for coord in coords:
                min_x, max_x = min(min_x, coord[0]), max(max_x, coord[0])
                min_y, max_y = min(min_y, coord[1]), max(max_y, coord[1])

            piece = np.zeros((max_x - min_x + 1, max_y - min_y + 1))

            for coord in coords:
                piece[coord[0] - min_x][coord[1] - min_y] = plabel

            sol.append((piece.astype(int), (min_x, min_y)))
        ### print(sol)
        return sol


def solveAll(board, pents):
    '''
    Function: solveAll
    Instruction: For mp2 part1, the solve function only need to return with one solution while 
    actually there could exit much more solutions. If we need to find all the solutions of the problem,
    simply using dfs could take too much spatial complexity and time complexity because it does a lot of 
    recursions. One optimization approach our team finds is call DLX algorithm, which can solve the
    problem with less O(n) and S(n) especially if with larger size of the board.
    '''
    # Part 1.1 Transform the polyomino problem into a exact cover problem


    # Part 1.2 using DLX algorithm to solve the exact cover problem
    # DLX: Using Dancing linked data structure to accomplish nondeterministic algorithm (X algorithm)
    # DLX is provided by Donald E. Knuth from Stanford University

    # Part 1.3 use the solution by the cover problem to get the return value 

