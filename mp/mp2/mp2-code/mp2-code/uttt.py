from time import sleep
from math import inf
from random import randint
import numpy as np

class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        self.maxPlayer='X'
        self.minPlayer='O'
        self.maxDepth=3
        #The start indexes of each local board
        self.globalIdx=[(0,0),(0,3),(0,6),(3,0),(3,3),(3,6),(6,0),(6,3),(6,6)]

        #Start local board index for reflex agent playing
        self.startBoardIdx=4
        #self.startBoardIdx=randint(0,8)

        #utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility=10000
        self.twoInARowMaxUtility=500
        self.preventThreeInARowMaxUtility=100
        self.cornerMaxUtility=30

        self.winnerMinUtility=-10000
        self.twoInARowMinUtility=-100
        self.preventThreeInARowMinUtility=-500
        self.cornerMinUtility=-30

        # cyh的参数
        self.Design_twoInARowMaxUtility=500
        self.Design_preventThreeInARowMaxUtility=100
        self.Design_twoInARowMinUtility=-100
        self.Design_preventThreeInARowMinUtility=-500
        self.centerMaxUtility = 80
        self.centerMinUtility = -80

        # 为了封装
        self.twoInARowMax = [self.twoInARowMaxUtility,self.Design_twoInARowMaxUtility]
        self.twoInARowMin = [self.twoInARowMinUtility,self.Design_twoInARowMinUtility]
        self.preventThreeInARowMax = [self.preventThreeInARowMaxUtility,self.Design_preventThreeInARowMaxUtility]
        self.preventThreeInARowMin = [self.preventThreeInARowMinUtility,self.Design_preventThreeInARowMinUtility]
        self.cornerMax = [self.cornerMaxUtility,self.centerMaxUtility]
        self.cornerMin = [self.cornerMinUtility,self.centerMinUtility]

        self.expandedNodes=0
        self.currPlayer=True

        # 初始化最佳落子位置
        self.bestIdx = (-1,-1)
        
        # IF design
        self.isDesign = False

#-------------------------- New Utility score Helper Function-------------------------------
    def ExtraRule(self, isMax):
        # 如果 two in a row 指向的 local board 依旧存在 two in a row， 则加20分。

        player = self.maxPlayer if isMax else self.minPlayer
        extra = 20 if isMax else -20

        utility = 0

        local_two = {}
        # Check if the player has two in a row in some local boards.
        
        for i, j in self.globalIdx:
            local_two[(i,j)]=False
            if (self.board[i][j] == player and self.board[i+1][j+1] == player and self.board[i+2][j+2] == '_') or (self.board[i][j] == player and self.board[i+1][j+1] == '_' and self.board[i+2][j+2] == player) or (self.board[i][j] == '_' and self.board[i+1][j+1] == player and self.board[i+2][j+2] == player):
                local_two[(i,j)] = True
                continue
            if (self.board[i][j+2] == player and self.board[i+1][j+1] == player and self.board[i+2][j] == '_') or (self.board[i][j+2] == player and self.board[i+1][j+1] == '_' and self.board[i+2][j] == player) or (self.board[i][j+2] == '_' and self.board[i+1][j+1] == player and self.board[i+2][j] == player):
                local_two[(i,j)] = True
                continue            
            for k in range(3):
                if (self.board[i+k][j] == player and self.board[i+k][j+1] == player and self.board[i+k][j+2] == '_') or (self.board[i+k][j] == player and self.board[i+k][j+1] == '_' and self.board[i+k][j+2] == player) or (self.board[i+k][j] == '_' and self.board[i+k][j+1] == player and self.board[i+k][j+2] == player):
                    local_two[(i,j)] = True
                    break
                if (self.board[i][j+k] == player and self.board[i+1][j+k] == player and self.board[i+2][j+k] == '_') or (self.board[i][j+k] == player and self.board[i+1][j+k] == '_' and self.board[i+2][j+k] == player) or (self.board[i][j+k] == '_' and self.board[i+1][j+k] == player and self.board[i+2][j+k] == player):
                    local_two[(i,j)] = True
                    break
                
        # check the corresponding local board
        for i, j in self.globalIdx:
            if self.board[i][j] == player and self.board[i+1][j+1] == player and self.board[i+2][j+2] == '_' and local_two[(6,6)]:
                utility += extra
            if self.board[i][j] == player and self.board[i+1][j+1] == '_' and self.board[i+2][j+2] == player and local_two[(3,3)]:
                utility += extra
            if self.board[i][j] == '_' and self.board[i+1][j+1] == player and self.board[i+2][j+2] == player and local_two[(0,0)]:
                utility += extra

            if self.board[i][j+2] == player and self.board[i+1][j+1] == player and self.board[i+2][j] == '_' and local_two[(6,0)]:
                utility += extra
            if self.board[i][j+2] == player and self.board[i+1][j+1] == '_' and self.board[i+2][j] == player and local_two[(3,3)]:
                utility += extra
            if self.board[i][j+2] == '_' and self.board[i+1][j+1] == player and self.board[i+2][j] == player and local_two[(0,6)]:
                utility += extra

            for k in range(3):
                if self.board[i+k][j] == player and self.board[i+k][j+1] == player and self.board[i+k][j+2] == '_' and local_two[(k*3,6)]:
                    utility += extra
                if self.board[i+k][j] == player and self.board[i+k][j+1] == '_' and self.board[i+k][j+2] == player and local_two[(k*3,3)]:
                    utility += extra
                if self.board[i+k][j] == '_' and self.board[i+k][j+1] == player and self.board[i+k][j+2] == player and local_two[(k*3,0)]:
                    utility += extra

                if self.board[i][j+k] == player and self.board[i+1][j+k] == player and self.board[i+2][j+k] == '_' and local_two[(6,k*3)]:
                    utility += extra
                if self.board[i][j+k] == player and self.board[i+1][j+k] == '_' and self.board[i+2][j+k] == player and local_two[(3,k*3)]:
                    utility += extra
                if self.board[i][j+k] == '_' and self.board[i+1][j+k] == player and self.board[i+2][j+k] == player and local_two[(0,k*3)]:
                    utility += extra

        return utility
    
    def SecondRule(self, isMax, designed=False):
        """
        This function calculates the utility score by the predifined second rule.

        designed: True(1) for designed agents; False(0) for predefined agents.
        """
        player = self.maxPlayer if isMax else self.minPlayer
        opponent = self.minPlayer if isMax else self.maxPlayer

        utility = 0

        # Check if the player has two in a row
        twoInARow = self.twoInARowMax[designed] if isMax else self.twoInARowMin[designed]
        for i, j in self.globalIdx:
            if (self.board[i][j] == player and self.board[i+1][j+1] == player and self.board[i+2][j+2] == '_') or (self.board[i][j] == player and self.board[i+1][j+1] == '_' and self.board[i+2][j+2] == player) or (self.board[i][j] == '_' and self.board[i+1][j+1] == player and self.board[i+2][j+2] == player):
                utility += twoInARow
            if (self.board[i][j+2] == player and self.board[i+1][j+1] == player and self.board[i+2][j] == '_') or (self.board[i][j+2] == player and self.board[i+1][j+1] == '_' and self.board[i+2][j] == player) or (self.board[i][j+2] == '_' and self.board[i+1][j+1] == player and self.board[i+2][j] == player):
                utility += twoInARow
            for k in range(3):
                if (self.board[i+k][j] == player and self.board[i+k][j+1] == player and self.board[i+k][j+2] == '_') or (self.board[i+k][j] == player and self.board[i+k][j+1] == '_' and self.board[i+k][j+2] == player) or (self.board[i+k][j] == '_' and self.board[i+k][j+1] == player and self.board[i+k][j+2] == player):
                    utility += twoInARow
                if (self.board[i][j+k] == player and self.board[i+1][j+k] == player and self.board[i+2][j+k] == '_') or (self.board[i][j+k] == player and self.board[i+1][j+k] == '_' and self.board[i+2][j+k] == player) or (self.board[i][j+k] == '_' and self.board[i+1][j+k] == player and self.board[i+2][j+k] == player):
                    utility += twoInARow


        # Check if the opponent has two in a row and the player can prevent it
        preventThreeInARow = self.preventThreeInARowMax[designed] if isMax else self.preventThreeInARowMin[designed]
        for i, j in self.globalIdx:
            if (self.board[i][j] == opponent and self.board[i+1][j+1] == opponent and self.board[i+2][j+2] == player) or (self.board[i][j] == opponent and self.board[i+1][j+1] == player and self.board[i+2][j+2] == opponent) or (self.board[i][j] == player and self.board[i+1][j+1] == opponent and self.board[i+2][j+2] == opponent):
                utility +=  preventThreeInARow
            if (self.board[i][j+2] == opponent and self.board[i+1][j+1] == opponent and self.board[i+2][j] == player) or (self.board[i][j+2] == opponent and self.board[i+1][j+1] == player and self.board[i+2][j] == opponent) or (self.board[i][j+2] == player and self.board[i+1][j+1] == opponent and self.board[i+2][j] == opponent):
                utility +=  preventThreeInARow
            for k in range(3):
                if (self.board[i+k][j] == opponent and self.board[i+k][j+1] == opponent and self.board[i+k][j+2] == player) or (self.board[i+k][j] == opponent and self.board[i+k][j+1] == player and self.board[i+k][j+2] == opponent) or (self.board[i+k][j] == player and self.board[i+k][j+1] == opponent and self.board[i+k][j+2] == opponent):
                    utility +=  preventThreeInARow
                if (self.board[i][j+k] == opponent and self.board[i+1][j+k] == opponent and self.board[i+2][j+k] == player) or (self.board[i][j+k] == opponent and self.board[i+1][j+k] == player and self.board[i+2][j+k] == opponent) or (self.board[i][j+k] == player and self.board[i+1][j+k] == opponent and self.board[i+2][j+k] == opponent):
                    utility +=  preventThreeInARow
        return utility
    

    def ThirdRule(self, isMax, designed=False):
        """
        This function calculates the score utility by the predifined second rule.

        designed: True(1) for designed agents; False(0) for predefined agents.
        """
        player = self.maxPlayer if isMax else self.minPlayer

        utility = 0
        for i, j in self.globalIdx:
            if self.board[i][j] == player:
                if isMax:
                    utility += self.cornerMax[designed]
                else:
                    utility += self.cornerMin[designed]
            if self.board[i+2][j] == player:
                if isMax:
                    utility += self.cornerMax[designed]
                else:
                    utility += self.cornerMin[designed]            
            if self.board[i][j+2] == player:
                if isMax:
                    utility += self.cornerMax[designed]
                else:
                    utility += self.cornerMin[designed]
            if self.board[i+2][j+2] == player:
                if isMax:
                    utility += self.cornerMax[designed]
                else:
                    utility += self.cornerMin[designed]
        return utility
    
    def DesignFourthRule(self, isMax):
        utility = 0
        num_center_occupy = 0
        player = self.maxPlayer if isMax else self.minPlayer

        for x, y in self.globalIdx:
            if self.board[x + 1][y + 1] == player:
                num_center_occupy += 1
        if self.isDesign:
            utility +=  num_center_occupy**1 * 60
        return utility

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]])+'\n')

    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0
        # First rule: 
        # Check if there is a winner
        winner = self.checkWinner()
        # If the offensive agent wins (form three-in-a-row)
        if winner == 1:
            return self.winnerMaxUtility
        elif winner == -1:
            return self.winnerMinUtility

        # Second rule:
        score += self.SecondRule(isMax)

        # Third rule: For each corner taken by the offensive agent, increment the utility
        score += self.ThirdRule(isMax)

        return score

    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0
        # Basic score
        # First rule: Check if there is a winner
        winner = self.checkWinner()
        # If the offensive agent wins (form three-in-a-row)
        if winner == 1:
            return self.winnerMaxUtility
        elif winner == -1:
            return self.winnerMinUtility
        # Second rule:two in low
        score += self.SecondRule(isMax, True)
        # Third rule: For each corner taken by the offensive agent, increment the utility
        score += self.ThirdRule(isMax, True)

        # Additional score
        score += self.DesignFourthRule(isMax)

        ### score += self.ExtraRule(isMax)

        return score

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        #YOUR CODE HERE
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == '_':
                    return True
        return False

        #movesLeft=True
        #return movesLeft

    def checkWinner(self):
        #Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        #YOUR CODE HERE
        p = self.maxPlayer
        for i,j in self.globalIdx:
                if self.board[i][j] == p and self.board[i+1][j+1] == p and self.board[i+2][j+2] == p:
                    return 1
                if self.board[i][j+2] == p and self.board[i+1][j+1] == p and self.board[i+2][j] == p:
                    return 1
                for k in range(3):
                    if self.board[i+k][j] == p and self.board[i+k][j+1] == p and self.board[i+k][j+2] == p:
                        return 1
                    if self.board[i][j+k] == p and self.board[i+1][j+k] == p and self.board[i+2][j+k] == p:
                        return 1

        p = self.minPlayer
        for i,j in self.globalIdx:
                if self.board[i][j] == p and self.board[i+1][j+1] == p and self.board[i+2][j+2] == p:
                    return -1
                if self.board[i][j+2] == p and self.board[i+1][j+1] == p and self.board[i+2][j] == p:
                    return -1
                for k in range(3):
                    if self.board[i+k][j] == p and self.board[i+k][j+1] == p and self.board[i+k][j+2] == p:
                        return -1
                    if self.board[i][j+k] == p and self.board[i+1][j+k] == p and self.board[i+2][j+k] == p:
                        return -1

        return 0

    def alphabeta(self,depth,currBoardIdx,alpha,beta,isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #design_
        #YOUR CODE HERE
        self.expandedNodes += 1
        #检查游戏是否结束或达到搜索深度
        if depth == 0 or self.checkWinner() != 0:
            if self.isDesign and isMax and depth%2==0:  
                return self.evaluateDesigned(isMax)
            return self.evaluatePredifined(isMax)

        # 如果当前是最大化玩家的回合
        if isMax:
            max_score = -inf
            x, y = self.globalIdx[currBoardIdx] # 判断落子区域

            # 考虑如果区域内无子可下，以至于要去全域落子的情况
            # if 
            # availableBoard = (i,j) for i in range(9) for j in range(9)


            for (i,j) in ((i,j) for i in range(3) for j in range(3)): 
            # 这是学的cyh说的generator的for循环，跑不通就问他！
                if self.board[x+i][y+j] == '_': # 尝试所有可以落子的位置
                    self.board[x+i][y+j] = self.maxPlayer # 落子
                    score = self.alphabeta(depth-1, i*3+j, alpha, beta, False) # 获取对方回合分数
                    self.board[x+i][y+j] = '_' # 复原
                    # max_score = max(max_score, score) # 更新最大值
                    if score > max_score:
                        max_score = score
                        if depth == 3: # 如果是根节点，输出落子位置
                            self.bestIdx = (x+i,y+j)
                    alpha = max(alpha, max_score) # 更新alpha值
                    if beta <= alpha: # alpha-beta剪枝
                        break
            return max_score

        # 如果当前是最小化玩家的回合，下面同理
        else:
            min_score = inf
            x, y = self.globalIdx[currBoardIdx]
            for (i,j) in ((i,j) for i in range(3) for j in range(3)):
                if self.board[x+i][y+j] == '_': 
                    self.board[x+i][y+j] = self.minPlayer
                    score = self.alphabeta(depth-1, i*3+j, alpha, beta, True)
                    self.board[x+i][y+j] = '_'
                    # min_score = min(min_score, score) 
                    if score < min_score:
                        min_score = score
                        if depth == 3: # 如果是根节点，输出落子位置
                            self.bestIdx = (x+i,y+j)
                    beta = min(beta, min_score) # 更新beta值
                    if beta <= alpha: # alpha-beta剪枝
                        break
            return min_score        

        #bestValue=0.0
        #return bestValue

    #def possible(self, currBoardIdx): #可以下的位置


    def minimax(self, depth, currBoardIdx, isMax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        # YOUR CODE HERE
        self.expandedNodes += 1
        # 检查游戏是否结束或达到搜索深度
        if depth == 0 or self.checkWinner() != 0:
            return self.evaluatePredifined(isMax)
        
        # 如果当前是最大化玩家的回合
        if isMax:
            max_score = -inf
            x, y = self.globalIdx[currBoardIdx] # 判断落子区域
            # 注意！我现在还没有考虑区域内无子可下，以至于要去全域落子的情况，后面再补充。
            for (i,j) in ((i,j) for i in range(3) for j in range(3)):
                if self.board[x+i][y+j] == '_': # 尝试所有可以落子的位置
                    self.board[x+i][y+j] = self.maxPlayer # 落子
                    score = self.minimax(depth-1, i*3+j, False) # 获取对方回合分数
                    self.board[x+i][y+j] = '_' # 复原
                    # max_score = max(max_score, score) # 更新最大值
                    if score > max_score:
                        max_score = score
                        if depth == 3: # 如果是根节点，输出落子位置
                            self.bestIdx = (x+i,y+j)                    
            return max_score

        # 如果当前是最小化玩家的回合,同理
        else:
            min_score = inf
            x, y = self.globalIdx[currBoardIdx]
            for (i,j) in ((i,j) for i in range(3) for j in range(3)):
                if self.board[x+i][y+j] == '_': 
                    self.board[x+i][y+j] = self.minPlayer
                    score = self.minimax(depth-1, i*3+j, True)
                    self.board[x+i][y+j] = '_'
                    # min_score = min(min_score, score) 
                    if score < min_score:
                        min_score = score
                        if depth == 3: # 如果是根节点，输出落子位置
                            self.bestIdx = (x+i,y+j)                    
            return min_score


        #bestValue=0.0
        #return bestValue


    def playGamePredifinedAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxDefensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        now=-1
        if(maxFirst):
            now=1 
        bestMove=[]
        bestValue=[]
        gameBoards=[]
        expandedNodes=[]
        winner=0
        currBoardIdx=self.startBoardIdx
        while(self.checkWinner()==0 and self.checkMovesLeft()):
            Value=0
            self.expandedNodes=0
            if(now==1):
                if(isMinimaxOffensive):
                    Value=self.minimax(3, currBoardIdx,True)
                else:
                    Value=self.alphabeta(3,currBoardIdx,-inf,inf,True)
            else:
                if(isMinimaxDefensive):
                    Value=self.minimax(3, currBoardIdx,False)
                else:
                    Value=self.alphabeta(3,currBoardIdx,-inf,inf,False)
            if(now==1):
                self.board[self.bestIdx[0]][self.bestIdx[1]]=self.maxPlayer
            else:
                self.board[self.bestIdx[0]][self.bestIdx[1]]=self.minPlayer
            gameBoards.append(self.board)
            bestMove.append(self.bestIdx)
            bestValue.append(Value)
            expandedNodes.append(self.expandedNodes-1)
            now*=-1
            currBoardIdx=self.bestIdx[0]%3*3+self.bestIdx[1]%3
            # print(self.bestIdx,currBoardIdx)
        winner=self.checkWinner()
        
        return gameBoards, bestMove, expandedNodes, bestValue, winner

    def playGameYourAgent(self, AgentFirst,boardID):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        self.isDesign = True
        #now=randint(0,1) 
        now = AgentFirst

        # print('now is ',now)
        if(now==0):
            now=-1      #-1 means our agent.
        bestMove=[]
        gameBoards=[]
        winner=0
        ###currBoardIdx=randint(0,8)
        currBoardIdx=boardID
        while(self.checkWinner()==0 and self.checkMovesLeft()):
            
            if(now==1):
                self.alphabeta(3,currBoardIdx,-inf,inf,True)
            else:
                self.isDesign = True
                self.alphabeta(3,currBoardIdx,-inf,inf,False)

            if(now==1):
                self.board[self.bestIdx[0]][self.bestIdx[1]]=self.maxPlayer
            else:
                self.board[self.bestIdx[0]][self.bestIdx[1]]=self.minPlayer
            gameBoards.append(self.board)
            bestMove.append(self.bestIdx)
            now*=-1
            currBoardIdx=self.bestIdx[0]%3*3+self.bestIdx[1]%3

        winner=self.checkWinner()
        self.isDesign = False
        return gameBoards, bestMove, winner

    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for our agent is the winner, -1 for the human is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        self.isDesign = False
        now=randint(0,1) 
        if(now==0):
            now=-1      #-1 means our agent.
        bestMove=[]
        gameBoards=[]
        winner=0
        currBoardIdx=randint(0,8)
        while(self.checkWinner()==0 and self.checkMovesLeft()):
            self.expandedNodes=0
            if(now==1):
                p=-1
                row=-1
                col=-1
                self.printGameBoard()
                while(p==-1):
                    print('Please put in Board #',currBoardIdx)
                    while(p==-1):
                        row=int(input('Please type in the row number(0-8): '))
                        if(row//3==currBoardIdx//3):
                            break
                        print('Invalid row number. ')
                    while(p==-1):
                        col=int(input('Please type in the column number(0-8): '))
                        if(col//3==currBoardIdx%3):
                            break
                        print('Invalid column number. ')
                    if(self.board[row][col]=='_'):
                        self.board[row][col]=self.maxPlayer
                        break
                    print('Invalid index. ')
                bestMove.append((row,col))
                currBoardIdx=row%3*3+col%3
            else:
                self.alphabeta(3,currBoardIdx,-inf,inf,False)
                self.board[self.bestIdx[0]][self.bestIdx[1]]=self.minPlayer
                currBoardIdx=self.bestIdx[0]%3*3+self.bestIdx[1]%3
                bestMove.append(self.bestIdx)

            gameBoards.append(self.board)
            now*=-1
            
        winner=self.checkWinner()
        if(winner==1):
            print('Congratulations, you Win!')
        elif(winner==-1):
            print('Unfortunately, you lose!')
        return gameBoards, bestMove, winner




if __name__=="__main__":
    # feel free to write your own test code
    #----------------------------Test for PredifinedAgent-----------------------------------
    '''
    uttt=ultimateTicTacToe()
    gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(False,True,False)
    #print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[-1][:3]])+'\n')
    #print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[-1][3:6]])+'\n')
    #print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[-1][6:9]])+'\n')
    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")
    '''
    
    #----------------------------Test for YourAgent----------------------------------------
    ### uttt=ultimateTicTacToe()
    total = 18
    Dwin = 0 # the number of times that the designed agent wins
    for i in range(9):
        for j in range(2):
            uttt=ultimateTicTacToe()
            gameBoards, bestMove, winner=uttt.playGameYourAgent(j, i)
            if winner == -1:
                Dwin += 1
         # print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[-1][:3]])+'\n')
         # print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[-1][3:6]])+'\n')
         # print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[-1][6:9]])+'\n')
    print("The number of times Your Agent win is",Dwin)
    print("The rate that your agent win is",Dwin/total)

    
    
    #----------------------------Test for human--------------------------------------------
    '''
    gameBoards, bestMove, winner=uttt.playGameHuman()
    print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[-1][:3]])+'\n')
    print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[-1][3:6]])+'\n')
    print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[-1][6:9]])+'\n')
    '''
