# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util



class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
       # print "Starting up MDPAgent!"
        #name = "Pacman"
        self.pacman_steps = list()
        self.reward_map = dict()
        self.util_map =dict()
        self.flag =0
        
    
    def final(self,state):
        self.pacman_steps = list()
        self.reward_map = dict()
        self.util_map =dict()
        self.flag =0

   
    def getAction(self, state):
    	  #calls the legal Actions api which returns a list of legal moves from Pac-man agents current location	
        legal = api.legalActions(state)
        #corners API returns a list of the four corners in each maze    
        corners = api.corners(state)
        #calls the API which returns wall locations 
        wall_location = api.walls(state)
        #print('walls', wall_location)
        #API call returns Pac-man agent's current location
        my_location = api.whereAmI(state)
        #API call returns food location
        food_location = api.food(state)
        #API call returns ghost locations 
        ghost_location = api.ghosts(state)
        #width of the maze
        self.width = (corners[3][0] - corners[0][0]) 
        #height of the maze
        self.height = (corners[3][1] - corners[1][1])

        #list of coordinates not in walls
        coordinates =[]
        for i in range(self.width):
            for j in range(self.height):
                node = (i,j)
                if node not in wall_location:
                    coordinates.append(node)
        

        #empty list which will contain the representation of the pac-man grid
        reward_grid = []
        utility_grid = []
        #populate the grid for the pacman maze dependant on the dimensions of the maze with 0's
        self.makegrid(self.height,self.width,reward_grid)
        self.makegrid(self.height,self.width,utility_grid)
        
        #set walls in the reward grid and the utility grid
        self.setwalls(reward_grid,wall_location)
        self.setwalls(utility_grid,wall_location)


        for node in coordinates:
            if node in food_location and (node not in ghost_location):
               self.reward_map[node]=10.0
            elif node in ghost_location:
                self.reward_map[node]=-1000.0
            else:
                self.reward_map[node]=0.0
        
        for node  in self.reward_map:
            x = node[0]
            y = node[1]

            self.setValue(reward_grid,x,y,self.reward_map[node])
        
        #gives initial values to the utility dict for the rewards used in the calculation of utilities
        self.util_map = self.reward_map


       
        count =1 
        #while loop represents the number of iterations of the bellman equation carried out
        while count <=3:
            #loop through the map coordinates and calculate a utility for each coordinate 
            for node in self.util_map:
                #get list of neighbour nodes for each coordinate, these nodes are used to get the expected utility values
                neighbours = self.children(node)
                
                up = self.expected_up(node, wall_location)
                down = self.expected_down(node,wall_location)
                left = self.expected_left(node,wall_location)
                right = self.expected_right(node,wall_location)

                expected_utilities = self.list_append(up,down,left,right)
                #if length of ghost is 1, signifies we are in the small map
                if len(ghost_location) == 1:
                    #assign distance to ghost to the variable below
                    dist_to_ghost = util.manhattanDistance(node,ghost_location[0])
                    #conditional statement alters the reward associated with a node dependant on its proximity to the ghost
                    if (dist_to_ghost >= 4):
                        gamma=0.8
                        self.util_map[node] = round( (gamma * max(expected_utilities)),2)
                    elif (dist_to_ghost <4) and (dist_to_ghost > 1):
                        gamma = 0.8
                        #assign the reward a -0.30 value if it is between 4 and 2 distance to fghost
                        self.util_map[node] = round(-0.30 + (gamma * max(expected_utilities)),2)
                    elif (dist_to_ghost ==1):
                        gamma = 0.8
                        #assign the reward -0.6 if it is 1 away from ghost
                        self.util_map[node] = round(-0.60 + (gamma * max(expected_utilities)),2)
                    elif (dist_to_ghost ==0):
                        #assign a reward of the ghost reward to the utility of where the ghost is 
                        self.util_map[node] = self.reward_map[node]
                    
                elif len(ghost_location)==2:

                    dist_to_ghost_one = util.manhattanDistance(node,ghost_location[0])
                    dist_to_ghost_two = util.manhattanDistance(node,ghost_location[1])
                
                    if (dist_to_ghost_one >= 4) and (dist_to_ghost_two >=4):
                        #gamma was  1 
                        gamma=1
                        self.util_map[node] = round( (gamma * max(expected_utilities)),2)
                    elif (dist_to_ghost_one <4) or (dist_to_ghost_two <4):
                        #0.6 appears superior marginally
                        gamma = 0.6
                        self.util_map[node] = round(-1.0 + (gamma * max(expected_utilities)),2)
                    elif (dist_to_ghost_one ==0) or (dist_to_ghost_two==0):
                        self.util_map[node] = self.reward_map[node]
                
                x = node[0]
                y = node[1]
                #use the set value function taken from Simon Parsons lab solution for getting maze
                self.setValue(utility_grid,x,y, self.util_map[node])
                
            
           
            #increase count so as to progress through the number of iterations the bellman update carries out before agent moves
            count = count +1
        
        
        #returns dictionary of moves available and the values each move has i.e (9,1):10
        move_dict = self.maximum_move(my_location,wall_location,legal)
    
        #returns the key (coordinate) with the greatest value
        max_coord = self.get_max_key(move_dict)

        #returns the direction for the maximum coord and the move needed for Pac-man to go there
        direction = self.pacman_moves(max_coord, my_location, legal)
        
        #return the api.makeMove
        return api.makeMove(direction,legal)



    def makegrid(self,height,width,grid):
        """ takes an empty list and appends 0's to the list in the dimensions of the grid. Taken from Lab 2 Solution """
        subgrid =[]
        for i in range(self.height +1):
            row=[]
            for j in range(self.width+1):
                row.append(0)
            grid.append(row)


    def setValue(self,grid,x,y,value):
        """ sets values in the grid, augmented from Lab 2 solution"""
        grid[y][x]=value

    def prettyDisplay(self,grid):   
        """ displays the grid taken from solution to Lab 2 solution"""
        for i in range(self.height+1):
            for j in range(self.width+1):
                print grid[self.height - (i )][j],
            print
        print
    
    def setwalls(self,grid,wall_location):
        """ sets walls for the grid"""
        for wall in wall_location:
            x = wall[0]
            y = wall[1]
            self.setValue(grid,x,y,'####')

    def children(self, node):
        """ returns the adjacent nodes to pacman including those in the wall"""
        children = []
        x,y = node
        north = (x, y+1)
        south = (x, y-1)
        west = (x-1,y)
        east = (x+1, y)
        
        children.append(north)
        children.append(south)
        children.append(west)
        children.append(east)
       
        return children

    def makeablemoves(self, node, wall_location):
        """ returns the makeable moves in a list, from pacmans current location"""
        children = []
        x,y = node
        north = (x, y+1)
        south = (x, y-1)
        west = (x-1,y)
        east = (x+1, y)
        
        if north not in wall_location:
            children.append(north)
        else:
            pass
        
        if south not in wall_location:
            children.append(south)
        else:
            pass
        if west not in wall_location:
            children.append(west)
        else:
            pass
        if east not in wall_location:
            children.append(east)
        else:
            pass
       
        return children

    
    def expected_up(self, node, wall_location):
        """ Calculates the expected utility if the agent attempted to move up"""
        neighbours = self.children(node)
        up = neighbours[0]
        left = neighbours[2]
        right = neighbours[3]
        
        if up not in wall_location:
            up = round((self.util_map[up] *0.8),2)
        else:
            up = round((self.util_map[node]*0.8),2)
        
        if left not in wall_location:
            left = round((self.util_map[left] *0.1),2)
        else:
            left= round((self.util_map[node] *0.1),2)
        
        if right not in wall_location:
            right = round((self.util_map[right] *0.1),2)
        else:
            right = round((self.util_map[node] *0.1),2)
        
        final_util = round((up + left + right),2)
        return final_util
    
    def expected_down(self, node, wall_location):
        """ Calculates the expected utility if the agent attempted to move down"""
        neighbours = self.children(node)
        down = neighbours[1]
        left = neighbours[2]
        right = neighbours[3]

        if down not in wall_location:
            down = round((self.util_map[down] *0.8),2)
        else:
            down = round((self.util_map[node]*0.8),2)
        
        if left not in wall_location:
            left = round((self.util_map[left] *0.1),2)
        else:
            left= round((self.util_map[node] *0.1),2)
        
        if right not in wall_location:
            right = round((self.util_map[right] *0.1),2)
        else:
            right = round((self.util_map[node] *0.1),2)
        
        final_util = round((down + left + right),2)
        return final_util
    
    def expected_left(self, node, wall_location):
        """ Calculates the expected utility if the agent attempted to move to the left"""
        neighbours = self.children(node)
        left = neighbours[2]
        up = neighbours[0]
        down = neighbours[1]

        if left not in wall_location:
            left = round((self.util_map[left] *0.8),2)
        else:
            left = round((self.util_map[node]*0.8),2)
        
        if up not in wall_location:
            up = round((self.util_map[up] *0.1),2)
        else:
            up= round((self.util_map[node] *0.1),2)
        
        if down not in wall_location:
            down = round((self.util_map[down] *0.1),2)
        else:
            down = round((self.util_map[node] *0.1),2)
        
        final_util = round((left + up + down),2)
        return final_util

    def expected_right(self, node, wall_location):
        """ Calculates the expected utility if the agent attempted to move to the right"""
        neighbours = self.children(node)
        right= neighbours[3]
        up = neighbours[0]
        down = neighbours[1]

        if right not in wall_location:
            right = round((self.util_map[right] *0.8),2)
        else:
            right = round((self.util_map[node]*0.8),2)
        
        if up not in wall_location:
            up = round((self.util_map[up] *0.1),2)
        else:
            up= round((self.util_map[node] *0.1),2)
        
        if down not in wall_location:
            down = round((self.util_map[down] *0.1),2)
        else:
            down = round((self.util_map[node] *0.1),2)
        
        final_util = round((right + up + down),2)
        return final_util
    
    def list_append(self,a,b,c,d):
        """ returns a list of items entered"""
        target_list = []
        target_list.append(a)
        target_list.append(b)
        target_list.append(c)
        target_list.append(d)

        return target_list

    def maximum_move(self, my_location, wall_location,legal):
        """ Returns a list of makeable moves for Pac-man then selects the move with the highest 
        expected utility according to the bellman equation"""

        makeable_moves = self.makeablemoves(my_location, wall_location)

        print 'my location', my_location
        print 'list of moves possible', makeable_moves
        #empty dict to hold the utilities of the possible moves
        move_dict = dict()

        for move in makeable_moves:
            move_dict[move]= self.util_map[move]
        
        return move_dict
    
    def get_max_key(self,move_dict):
        'get the coordinate with the maximum utility for the next move from a dictionary of possible moves'
        list_of_values = move_dict.values()
        list_of_keys = move_dict.keys()
        index_of_key = list_of_values.index(max(list_of_values))
        max_coord = list_of_keys[index_of_key]
        return max_coord
    
    def pacman_moves(self,node, my_location, legal):
        'returns direction for pacman agent to traverse dependant on the adjacent node/coordinate with the highest utility'
        
        if my_location[0] < node[0]:
            if Directions.EAST in legal:
                return Directions.EAST
        
        elif my_location[0] > node[0]:
            if Directions.WEST in legal:
                return Directions.WEST
        
        elif my_location[1] > node[1]:
            if Directions.SOUTH in legal:
                return Directions.SOUTH
        
        elif my_location[1] < node[1]:
            if Directions.NORTH in legal:
                return Directions.NORTH

    
    
    

        
        
