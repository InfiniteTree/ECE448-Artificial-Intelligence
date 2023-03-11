# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

from collections import deque
import maze

path = [] # temp path that is recorded
result = [] # Final successful path
find_flag = False # Set a flag to indicate whether it is successfully to find a route
num_ep = 0 # Number of state has been explored
visited = [] # for dfs
visited_queue = [] # for bfs,greedy

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)


#----------------------Helper Function------------------------#
'''
Function init_search:
Instruction: Initialize the relative avgs that are needed for the dfs search functions
Input: maze
Output: None
Side effect: Relative avgs
'''
def init_search(maze):
    global path, result, find_flag, num_ep, start, end, visited, visited_queue, sequeue
    # --------------------------For dfs--------------------------------
    path = [] # temp path that is recorded
    result = [] # Final successful path
    find_flag = False # Set a flag to indicate whether it is successfully to find a route
    num_ep = 0 # Number of state has been explored
    maze_row = maze.getDimensions()[0]
    maze_col = maze.getDimensions()[1]
    start = maze.getStart()
    end = maze.getObjectives()[0]
    visited = [[False] * maze_col for i in range(maze_row)] # row*col matrix to indicated the visited state
    visited_queue = [] # for bfs,greedy
    # --------------------------For bfs, greedy--------------------------------
    sequeue = [] # Represent for searched state in queue


'''
Function init_astar_search:
Instruction: Initialize the relative avgs that are needed for the astar search functions
Input: maze
Output: None
Side effect: Relative avgs
'''
def init_astar_search(maze):
    global path, shortest_path, visited, start, end, Frontier,num_ep
    path = {}
    shortest_path = deque()
    visited = deque()
    start = maze.getStart()
    end = maze.getObjectives()[0]
    Frontier = deque()
    num_ep = 0


'''
Function get_Mids:
Instruction: Get the manhantan distance between the two points
Input: start, end
Output: the manhantan distance
'''
def get_Mdis(start, end):
    return abs(start[0]-end[0])+abs(start[1]-end[1])


'''
Function bfs
Instruction:
by the avg Frontier, we use FIFO to search for the possible next states
to find the path, we use a dictionary to store the all the paths that has been explored by BFS.
Once the object is stored as the value in the dictionary, use list.append to store the actual
shortest path in a list
Use queue to achieve the algorithm
Input: maze, start, end
Output: path, num_ep
Side Effect: Relative global avgs
'''
def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    path = {}
    shortest_path = deque()
    visited = deque()
    start_state = maze.getStart()
    objective = maze.getObjectives()[0]
    Frontier = deque()
    Frontier.append(start_state)
    # Fetch the first element of the Frontier to the head_state, head_state is the current state 
    # and remove it from the Frontier
    head_state = start_state
    # The first objective
    tail_state = head_state

    # While some states still exits in Frontier
    while Frontier:
        if head_state == objective:
            tail_state = head_state
            break
        if head_state in visited:
            # pop until head_state is the state has not been visited
            head_state = Frontier.popleft()
        
        if head_state not in visited:
            # newNeighbor can have maximumu size as [(x1,y1),(x2,y2),(...),(...)]
            newNeighbors = maze.getNeighbors(head_state[0], head_state[1])
            # ts stands for tail_state
            for ts in newNeighbors:
                if maze.isValidMove(ts[0],ts[1]) and ts not in visited:
                    # Using the head_state as the key value for key can not be repeated while value can be repeated in a dictionary
                    # path will look like this: {tail_state1: head_state, tail_state2: head_state, ... }
                    path[ts] = head_state
                    Frontier.append(ts)
            visited.append(head_state) 
            if head_state == objective:
                tail_state = head_state
                break            
    
    # Enable to search for [-1] in shortest_path
    shortest_path.append(tail_state)

    while shortest_path[0] != start_state:        
        shortest_path.appendleft(path[tail_state])
        tail_state = path[tail_state]
    
    return list(shortest_path), len(path)


def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    # Data initilization
    init_search(maze)

    # Call the recursive function
    dfs_search(maze,start,end)
    ### print("number of state explored by dfs:",num_ep)
    return result, num_ep

'''
Function dfs_search
Instruction:
Define a search function for dfs that can accept the start and the end avgs 
for recursion function usage. in dfs we search the route by randomly choose a
neighbor state to explore until we find the objective By dfs the time/cost(num_ep) 
to get the objective depends a lot on the order of the recursion direction order.
Use stack to achieve the algorithm
Input: maze, start, end
Output: path, num_ep
Side Effect: Relative global avgs
'''
def dfs_search(maze,start,end):
    global find_flag, visited, path, result, num_ep
    x,y=start
    # print("\nThe visited[x][y] is",visited[x][y])
    if not find_flag and maze.isValidMove(x,y) and visited[x][y] is False:
        visited[x][y] = True
        num_ep += 1 # number of state has been explored increase

        if start == end: # The objective is found
            result = path.copy()+[end]
            find_flag = True
            return # return immediately after a route is found
        path.append((x,y))

        # Recursion to find the objective
        dfs_search(maze,(x+1,y),end)
        dfs_search(maze,(x,y+1),end)
        dfs_search(maze,(x-1,y),end)
        dfs_search(maze,(x,y-1),end)
        
        path.pop() # BackTracking
        # Here return not needed


def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    # Initialization of relative return value and temp data
    init_search(maze)

    # Call the search function by greedy algorithm
    result = greedy_search(maze,[start],start)
    path = result[1]
    num_ep = len(visited_queue)
    return path, num_ep

'''
Function greedy_search
Instruction:
Define a search function for greedy that can accept the start and the final_queue_ avgs 
for recursion function usage. The greedy algorithm consider the manhantance distance as the 
heuristic number to determine the recursion order in dfs to decide which direction order to 
do recursion first. Usually the solution produced by the greedy algorithm will be locally optimal
while may not be a globally optimal one.
Input: maze, start, end
Output: path, num_ep
Side Effect: Relative global avgs
'''
def greedy_search(maze, final_queue_,start):
    global visited, visited_queue, sequeue, result, num_ep
    x,y=start
    neighbors = maze.getNeighbors(x,y)
    sequeue = []

    for state in neighbors:
        sequeue.append({"state": state, "dist": get_Mdis(state, end)})
    sequeue = sorted(sequeue, key=lambda i: i['dist'])

    final_queue = final_queue_
    # Made the new visited queue
    visited_queue.append(start) 
    for state in sequeue:
        if not (state["state"] in visited_queue):
            # Traversal the new neighbors of the state
            # return with a True for convinient to judge in next recursion
            final_queue.append(state["state"])
            if maze.isObjective(state["state"][0], state["state"][1]):
                return [True, final_queue]
            # Recursion to get the final state
            result = greedy_search(maze, final_queue, state["state"])
            if result[0]:
                return [True,result[1]] 
            else:
                final_queue.remove(state["state"])
                pass
    return [False]

def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    # Astar for single Dots
    # Initialization
    init_astar_search(maze)

    if len(maze.getObjectives()) == 1:
        # Call the search function to find the single dot
        path, num_ep = astar_single(maze,start,end)
    elif len(maze.getObjectives()) > 1:
        path, num_ep = astar_multi(maze)
    return path, num_ep

''' 
Function astar_search
Instruction:
# c(n) = g(n) + d(n)
# f(n) = g(n) + h(h)
# c(n) is the actual cost to run from the current state to the goal and f(n) is the idea cost to run
# where g(n) is the cost that have already been taken, d(n) is the actual cost that should be take to 
# reach the goal, and h(n) is the heuristic number(in the maze we use the Mahanttan distance to the goal)
# By A* search we can try to find the smallest f(n) which can furthur reach the goal to be the state of 
# the shortest path
Input: maze, start, end
Output: path, num_ep
Side Effect: Relative global avgs
'''

def astar_single(maze,start,end):
    global path, shortest_path, visited, Frontier, num_ep
    inni_Mdis = get_Mdis(start, end) 
    Frontier.append([inni_Mdis,start])
    pre_Frontier = Frontier

    # Fetch the first element of the Frontier to the head_state, head_state is the current state 
    # and remove it from the Frontier
    head_state = start
    tail_state = start
    ### print("inni_Mdis is ", inni_Mdis)
    ### print("inni_Frontier is ", Frontier)
    # n represents g(n)
    n = 0
    while pre_Frontier:
        if head_state in visited:
            # pop until head_state is the state has not been visited
            head_state = Frontier.popleft()[1]
        if head_state not in visited:
            # newNeighbor can have maximumu size as [(x1,y1),(x2,y2),(...),(...)]
            num_ep += 1 # explore the state
            newNeighbors = maze.getNeighbors(head_state[0], head_state[1])
            # ts stands for tail_state
            for ts in newNeighbors:
                if maze.isValidMove(ts[0],ts[1]) and ts not in visited and ts not in Frontier:
                    # Using the head_state as the key value for key can not be repeated while value can be repeated in a dictionary
                    # path pattern: {tail_state1: head_state, tail_state2: head_state, ... }
                    path[ts] = head_state
                    # Manhattan distance is regraded as the heuristic number
                    Mdis = get_Mdis(ts, end)
                    # f(n) = n + Mdis
                    Frontier.append([n + Mdis,ts])
            visited.append(head_state)

            #!!!make a copy
            pre_Frontier = list(Frontier)
            if head_state == end:
                tail_state = head_state
                break
            head_state = Frontier.popleft()[1]
            n += 1
        # Sort the queue by the value of the fn and return it with the order from small fn to large fn
        # Therefore next loop we can first try to find the path starting at the state with smallest fn
            Frontier = sorted(Frontier, key = lambda x:x[0])
            Frontier = deque(Frontier)

        if head_state == end:
            tail_state = head_state
            break
        
    shortest_path.append(tail_state)
    
    while shortest_path[0] != start:        
        shortest_path.appendleft(path[tail_state])
        tail_state = path[tail_state]

    path = list(shortest_path)
    return path, num_ep

''' 
Function toNextP_Heuristic
Instruction: 
Use Prim algorithm to get the Minimum Spanning Tree
Input: start state, objectives remained to go
Output: an objective following the heuristic out of the whole list
'''

def toNextP_Heuristic(start,objectives_togo):
    ans=list()
    for i in objectives_togo:
        ans.append(abs(start[0]-i[0])+abs(start[1]-i[1]))
    threshold=min(ans)
    return threshold

# Astar for multiple objective dots
def astar_multi(maze):
    shortest_path=deque()
    start_state=maze.getStart()
    objectives=maze.getObjectives() #objective is a list
    
    count=0  # Counter to record the path founded
    num_ep = 0
    # Search until the all the objectives found
    while objectives:
        # fn = hn + gn
        # hn represents the MST, use Prim algorithm to get MST 
        # cn represents current cost to the Cur_Node
        visited=deque()
        unvisited = deque(objectives)
        
        for i in visited:
            unvisited.pop(i)
        fn, cn, hn = {}, {}, {}
        cn[start_state]=0
        hn[start_state]=toNextP_Heuristic(start_state,objectives) #+ MST(maze,start_state,unvisited)
        fn[start_state]=hn[start_state]+cn[start_state]
        Frontier=deque()
        Frontier.append(start_state)
        onePath=deque()
        parent={}
        # Search to get a path
        while Frontier:
            Node=Frontier.popleft()
            if Node in objectives:
                visited.append(Node)
                num_ep += 1
                objectives.remove(Node)
                Visited_objective = Node
                break    
            if Node in visited:
                Node=Frontier.popleft()

            if Node not in visited:
                visited.append(Node)
                num_ep += 1
                for i in maze.getNeighbors(Node[0],Node[1]):
                    if maze.isValidMove(i[0],i[1]) and i not in visited and i not in Frontier:
                        cur_hn=toNextP_Heuristic(i,objectives)
                        cur_cn=cn[Node]+1
                        if i in Frontier and fn[i]<cur_cn+cur_hn: # Jump back to for loop to check the next neighbor
                            continue
                        else: # Explore the state i
                            fn[i]=cur_hn+cur_cn
                            hn[i]=cur_hn
                            cn[i]=cur_cn
                            Frontier.append(i)
                            parent[i]=Node # Update the parent of i as Node


        onePath.appendleft(Visited_objective)

        # BackTracking one path to get the objective until the start state is found
        while start_state not in onePath:
            Visited_objective=parent[Visited_objective]
            onePath.appendleft(Visited_objective)
        start_state=visited[-1]
        
        # To find first Path, not need to delete the start_state
        if count==0:
            shortest_path+=onePath
        # For the next path to the objective , we need to delete the start_state
        else:
            onePath.remove(Visited_objective)
            shortest_path+=onePath
        count+=1
        ### print ("Count is ",count)
    return list(shortest_path),num_ep

# Remains to do
def extra():
    return [],0
    

