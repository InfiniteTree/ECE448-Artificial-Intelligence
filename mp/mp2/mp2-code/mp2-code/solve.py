# -*- coding: utf-8 -*-
import numpy as np
import Pentomino

'''
Function getPTypeAndPLable
Input: A pent
Output: Two ints: one represents the label number of the pent matrix and 
                the other represents the the number of squares that some pent is made by 
'''
def getPTypeAndLable(pent):
    PLabel, PType = 0, 0
    for row in range(pent.shape[0]):
        for col in range(pent.shape[1]):
            if pent[row][col] != 0:
                PType += 1
                if PLabel == 0:
                    PLabel = pent[row][col]
    return PType, PLabel

'''
Function getMutation
Input: A pent
Output: A list contain all of the possible mutations by flipping and rotation of a pent
        with the relative coordinates of each squares
'''
def getMutations(pent):
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

'''
Function getSingleMutation
Input: A mutation
Output: A tuple contain one mutations by flipping or rotation of a pent
        with the relative coordinates of each squares
'''
def getSingleMutation(mutation):
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

'''
Function getPentsMap
Input: Pents
Output: A dictionary contain the (type, label) as the key,
        and the coordinates of all mutations by all pents as the value
'''
def getPentsMap(pents):
    pentsMaps = {}
    p_type = 0 # (2/3/5) the number of squares that some pent is made by 
    p_label = 0 # for pento:1-12
    for pent in pents:
        p_type, p_label = getPTypeAndLable(pent)
        pentsMaps[p_type,p_label] = getMutations(pent)
    return pentsMaps


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
    # Part 1.1 Get the mapping of the pents
    # by judge the type(5/3/2) and the lable(1-12) of pents and determine all the mutations of each pent
    pents_maps = getPentsMap(pents)
    
    # Part 1.2 Precisely corver the board using dfs algorithm
    pi = 1

    pi += 1

    return 