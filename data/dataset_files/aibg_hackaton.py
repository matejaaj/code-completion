import sys
import time
import json
import random
from abc import *
from collections import deque

class GameState:
    def __init__(self, data):
        self.turn = int(data['turn'])
        self.firstPlayerTurn = bool(data['firstPlayerTurn'])
        self.player1 = Player(data['player1'])
        self.player2 = Player(data['player2'])
        self.board = [[str(cell) for cell in row] for row in data['board']]

class Player:
    def __init__(self, data):
        self.name = str(data['name'])
        self.energy = int(data['energy'])
        self.xp = int(data['xp'])
        self.coins = int(data['coins'])
        self.position = [int(pos) for pos in data['position']]
        self.increased_backpack_duration = str(data['increased_backpack_duration'])
        self.daze_turns = int(data['daze_turns'])
        self.frozen_turns = int(data['frozen_turns'])
        self.backpack_capacity = int(data['backpack_capacity'])
        self.raw_minerals = int(data['raw_minerals'])
        self.processed_minerals = int(data['processed_minerals'])
        self.raw_diamonds = int(data['raw_diamonds'])
        self.processed_diamonds = int(data['processed_diamonds'])
    

    def find_nearest_target(self, board, target_label):
        min_distance = float('inf')
        nearest_target_coords = None

        for i in range(len(board)):
            for j in range(len(board[i])):
                cell = board[i][j]
                if isinstance(cell, str) and cell.startswith(target_label):
                    distance = ((self.position[0] - i) ** 2 + (self.position[1] - j) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        nearest_target_coords = [i, j] 

        return nearest_target_coords

    def find_best_ore(self, board, target_label, value_weight=0.5):
        best_score = float('-inf')
        best_ore_coords = None

        for i in range(len(board)):
            for j in range(len(board[i])):
                cell = board[i][j]
                if isinstance(cell, str) and cell.startswith(target_label):
                    parts = cell.split('_')
                    if len(parts) >= 3 and int(parts[1]) >= 2:
                        number1 = int(parts[1])
                        number2 = int(parts[2])

                        distance = ((self.position[0] - i) ** 2 + (self.position[1] - j) ** 2) ** 0.5
                        score = number1 + value_weight / (distance + 0.1) 

                        if score > best_score:
                            best_score = score
                            best_ore_coords = [i, j]

        return best_ore_coords


    def find_accessible_neighbor(self, board, target_coords, player_coordinates):
        target_i, target_j = target_coords
        player_i, player_j = player_coordinates
        neighbors = [
            (target_i - 1, target_j), (target_i + 1, target_j),
            (target_i, target_j - 1), (target_i, target_j + 1)
        ]

        closest_neighbor = None
        min_distance = float('inf')  # Initialize min_distance with infinity

        for ni, nj in neighbors:
            if 0 <= ni < len(board) and 0 <= nj < len(board[0]):  # Check bounds
                if board[ni][nj] == 'E':  # Check if the neighbor is accessible
                    # Calculate Euclidean distance from the neighbor to the player's position
                    distance = ((ni - player_i) ** 2 + (nj - player_j) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_neighbor = [ni, nj]

        return closest_neighbor

    def BuildAction(self,x,y):
        return f'build {x} {y}'

    def RestAction():
        return 'rest'

    def ShopAction(self,action):
        return f'shop {action}'

    def AttackAction(self,x,y):
        return f'attack {x} {y}'

    def ConversionsAction(self,dCoins,mCoins,dEnergy,mEnergy,dXp,mXp):
        return f"conv {dCoins} diamond {mCoins} mineral to coins, {dEnergy} diamond {mEnergy} mineral to energy, {dXp} diamond {mXp} mineral to xp"

    def PutAction(self,X,Y,N,M):
        return f'refinement-take {X} {Y} mineral {N} diamond {M}'

    def TakeAction(self,X,Y,N,M):
        return f'refinement-take {X} {Y} mineral {N} diamond {M}'

    def ConversionXPSplit(self):
        total_minerals = self.raw_minerals
        total_diamonds = self.raw_diamonds 

        if self.energy <= 500:
            if total_minerals > 2:
                m_energy =  max(2, total_minerals - 2)
                m_xp = total_minerals - m_energy 
            else:
                m_energy = 0 
                m_xp = total_minerals 

            if total_diamonds >= 5:
                d_xp = max(total_diamonds, 5) 
                d_coins = 0
            else:
                d_coins = 0  
                d_xp = total_diamonds  

            m_coins = 0
            d_energy = 0
        else:
            if total_minerals > 2:
                m_xp =  max(2, total_minerals - 2)
                m_energy = total_minerals - m_xp
            else:
                m_energy = 0 
                m_xp = total_minerals 
            if total_diamonds >= 5:
                d_xp = max(total_diamonds, 5) 
                d_coins = total_diamonds - d_xp 
            else:
                d_coins = 0  
                d_xp = total_diamonds    

            m_coins = 0
            d_energy = 0

        return self.ConversionsAction(d_coins, m_coins, d_energy, m_energy, d_xp, m_xp)


    def GetDestroyFactorySequence(self, gameState):
        player_coordinates = self.GetPlayerCordinates(gameState)
        target_coordinates = self.find_nearest_target(gameState.board, 'F')
        if target_coordinates is None: return
        move_coordinates = self.find_accessible_neighbor(gameState.board, target_coordinates, player_coordinates)
        actions = self.get_move_sequence(gameState, player_coordinates, move_coordinates)

        factory = gameState.board[target_coordinates[0]][target_coordinates[1]]
        factory_parts = factory.split('_')
        counter = int(factory_parts[2])
        while(counter > 0 ):
            actions.append(f"attack {target_coordinates[0]} {target_coordinates[1]}")
            counter -= 1
        return actions


    def ConversionMoneySplit(self):
        total_minerals = self.raw_minerals
        total_diamonds = self.raw_diamonds 

        # Initialize variables with default values
        d_coins = 0
        d_xp = 0
        m_energy = 0
        m_xp = 0
        m_coins = 0
        d_energy = 0

        if self.energy <= 500:
            if total_minerals >= 2:
                m_energy = max(2, total_minerals - 2)
                m_xp = total_minerals - m_energy 
            else:
                m_xp = total_minerals 

            if total_diamonds >= 5:
                d_xp = total_diamonds
        else:
            if total_minerals >= 2:
                m_xp = max(2, total_minerals - 2)
                m_energy = total_minerals - m_xp
            else:
                m_xp = total_minerals 

            d_xp = total_diamonds
            d_coins = 0

        return self.ConversionsAction(d_coins, m_coins, d_energy, m_energy, d_xp, m_xp)


        
    def GetOreCapacity(self, board, coordinates):
        ore = board[coordinates[0]][coordinates[1]]
        parts = ore.split('_')
        return int(parts[1])
    
    def GetOreValue(self, board, coordinates):
        ore = board[coordinates[0]][coordinates[1]]
        parts = ore.split('_')
        return 2 if parts[0] == 'M' else 5
    
    def GetHomePosition(self, gameState):
        return [9,0] if gameState.firstPlayerTurn else [0,9]

    def GoHomeActions(self, gameState, player_coordinates, home_coordinates):

        actions = self.get_move_sequence(gameState, player_coordinates, home_coordinates)
        if actions is None: return ["shop daze"]
        return actions

    def GetPlayerCordinates(self, gameState):
        return myPlayer(gameState).position

    def GetMiningSequence(self, gameState, objectToMine):
        player_coordinates = self.GetPlayerCordinates(gameState)
        target_coordinates = self.find_best_ore(gameState.board, objectToMine)
        move_coordinates = self.find_accessible_neighbor(gameState.board, target_coordinates, player_coordinates)
        actions = self.get_move_sequence(gameState, player_coordinates,  move_coordinates)

        player_coordinates = move_coordinates
        player_capacity = int(myPlayer(gameState).backpack_capacity)  
        ore_capacity = self.GetOreCapacity(gameState.board, target_coordinates)
        ore_value = self.GetOreValue(gameState.board, target_coordinates)

        while player_capacity <= 8 and ore_capacity > 0:
            if player_capacity + ore_value > 8:
                break
            
            actions.append(f"mine {target_coordinates[0]} {target_coordinates[1]}")
            player_capacity += ore_value
            ore_capacity -= 1 
            
            self.raw_minerals += 1

        home_coordinates = self.GetHomePosition(gameState)

        actions.extend(self.GoHomeActions(gameState, player_coordinates, home_coordinates))
        actions.append(self.ConversionXPSplit())
        self.raw_minerals = 0
        return actions

    def GetMiningMoneySequence(self, gameState, objectToMine):
        player_coordinates = self.GetPlayerCordinates(gameState)
        target_coordinates = self.find_best_ore(gameState.board, objectToMine)
        move_coordinates = self.find_accessible_neighbor(gameState.board, target_coordinates, player_coordinates)
        actions = self.get_move_sequence(gameState, player_coordinates,  move_coordinates)

        player_coordinates = move_coordinates
        player_capacity = int(myPlayer(gameState).backpack_capacity)  
        ore_capacity = self.GetOreCapacity(gameState.board, target_coordinates)
        ore_value = self.GetOreValue(gameState.board, target_coordinates)

        while player_capacity <= 8 and ore_capacity > 0:
            if player_capacity + ore_value > 8:
                break
            
            actions.append(f"mine {target_coordinates[0]} {target_coordinates[1]}")
            player_capacity += ore_value
            ore_capacity -= 1 
            
            self.raw_minerals += 1

        home_coordinates = self.GetHomePosition(gameState)

        actions.extend(self.GoHomeActions(gameState, player_coordinates, home_coordinates))
        actions.append(self.ConversionMoneySplit())
        self.raw_minerals = 0
        return actions
    
    def get_move_sequence(self, gameState,  from_coordinates, target_coordinates):
        search = BreadthFirstSearch(gameState)
        initial_state = RobotState(gameState, None, from_coordinates, target_coordinates)

        path, _, _ = search.search(lambda: initial_state)

        if path is None:
            return []

        path.pop(0)
        return self.create_move_command(path)

    def create_move_command(self, actions):
        commands = []
        for action in actions:  
            commands.append(f"move {action[0]} {action[1]}")

        return commands

class Search(object):

    def __init__(self, gameState):
        self.board = gameState.board
        if gameState.firstPlayerTurn:
            self.player = gameState.player1
        else:
            self.player = gameState.player2

    def search(self, initial_state):

        initial_state = initial_state()  
        states_list = deque([initial_state]) 
        states_set = {initial_state.unique_hash()}  

        processed_list = deque([])  
        processed_set = set() 

        while len(states_list) > 0: 
            curr_state = self.select_state(states_list)  
            states_set.remove(curr_state.unique_hash())  

            processed_list.append(curr_state) 
            processed_set.add(curr_state.unique_hash())  

            if curr_state.is_final_state():  
                return Search.reconstruct_path(curr_state), processed_list, states_list

            new_states = curr_state.get_next_states()
            new_states = [new_state for new_state in new_states if
                          new_state.unique_hash() not in processed_set and
                          new_state.unique_hash() not in states_set]

            states_list.extend(new_states)

            states_set.update([new_state.unique_hash() for new_state in new_states])
        return None, processed_list, states_list

    @staticmethod
    def reconstruct_path(final_state):
        path = []
        while final_state is not None:
            path.append(final_state.position)
            final_state = final_state.parent
        return list(reversed(path))

    @abstractmethod
    def select_state(self, states):
        pass

class BreadthFirstSearch(Search):
    def select_state(self, states):
        return states.popleft()     
class State(object):

    @abstractmethod
    def __init__(self, gameState, parent=None, position=None, goal_position=None):

        self.board = gameState.board 
        self.parent = parent 
        self.gameState = gameState

        if self.parent is None:
            if gameState.firstPlayerTurn:
                  self.position = position
                  self.player_code = "1"
                  self.player = gameState.player2
            else:
                self.position = position
                self.player_code = "2"
                self.player = gameState.player2
            self.goal_position = goal_position  
        else: 
            self.position = position
            self.goal_position = goal_position
            self.player = self.parent.player
            self.player_code = self.parent.player_code

        self.depth = parent.depth + 1 if parent is not None else 1  

    def get_next_states(self):
        new_positions = self.get_legal_positions()
        next_states = []
        for new_position in new_positions:
            next_state = self.__class__(self.gameState, self, new_position, self.goal_position)
            next_states.append(next_state)
        return next_states

    def get_agent_code(self):
        return self.player_code

    @abstractmethod
    def get_legal_positions(self):
        """
        Apstraktna metoda koja treba da vrati moguce (legalne) sledece pozicije na osnovu trenutne pozicije.
        :return: list
        """
        pass

    @abstractmethod
    def is_final_state(self):
        """
        Apstraktna metoda koja treba da vrati da li je treuntno stanje zapravo zavrsno stanje.
        :return: bool
        """
        pass

    @abstractmethod
    def unique_hash(self):
        """
        Apstraktna metoda koja treba da vrati string koji je JEDINSTVEN za ovo stanje
        (u odnosu na ostala stanja).
        :return: str
        """
        pass
    
    
    @abstractmethod
    def get_current_cost(self):
        """
        Apstraktna metoda koja treba da vrati stvarnu dosadašnju trenutnu cenu za ovo stanje, odnosno g(n)
        Koristi se za vodjene pretrage.
        :return: float
        """
        pass
class RobotState(State):

    def __init__(self, gameState, parent=None, position=None, goal_position=None):
        super().__init__(gameState, parent, position, goal_position)
        if parent is None:
            self.cost = 0
        else: 
            self.cost = self.parent.cost + 1

    def get_legal_positions(self):
        retVal = []
        
        x = int(self.position[0])
        y = int(self.position[1])
        
        if self.get_agent_code == "1":
            base = "A"
        else:
            base = "B"

        available_fileds = [base, "E", self.get_agent_code]
        if y < 9:
            for i in range(y + 1, 10):
                if self.board[x][i] not in available_fileds:
                    break
                else:
                    retVal.append([x, i])

        if y > 0:
            for i in range(y - 1, -1, -1):
                if self.board[x][i] not in available_fileds:
                    break
                else:
                    retVal.append([x, i])

        if x < 9:
            for i in range(x + 1, 10):
                if self.board[i][y] not in available_fileds:
                    break
                else:
                    retVal.append([i, y])

        if x > 0:
            for i in range(x - 1, -1, -1):
                if self.board[i][y] not in available_fileds:
                    break
                else:
                    retVal.append([i, y])

        return retVal

    def is_final_state(self):
        return self.position == self.goal_position

    def unique_hash(self):
        return f"{self.position}{self.player.energy}{self.player.raw_minerals}"

    def manhattan_distance(self, pointA, pointB):
        return abs(pointA[0] - pointB[0]) + abs(pointA[1] - pointB[1])

    
    def get_current_cost(self):
        return self.cost

def count_letters_in_matrix(matrix):
    count_D = 0
    count_M = 0
    
    # Iterate through each row in the matrix
    for row in matrix:
        # Iterate through each element in the row
        for element in row:
            # Count the occurrences of 'D' and 'M' in the element
            count_D += element.count('D')
            count_M += element.count('M')

    return [count_D, count_M]



def ValidateDaze(gameState):
    player = myPlayer(gameState)
    if(player.coins >= 10 ): 
        return True
    else:
        return False

def is_enemy_nearby(x, y, enemy_position, board_size=10):
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),         (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    for dx, dy in neighbor_offsets:
        neighbor_x = x + dx
        neighbor_y = y + dy
        if 0 <= neighbor_x < board_size and 0 <= neighbor_y < board_size:
            if (neighbor_x, neighbor_y) == enemy_position:
                return True

    return False

move_sequence = []

cnt_money = 0
cnt_xp = 0
daze_cnt = 0
test_bot_cnt = 0

def myPlayer(gameState):
    if gameState.firstPlayerTurn:
        return gameState.player1
    else:
        return gameState.player2

def enemyPlayer(gameState):
    if gameState.firstPlayerTurn:
        return gameState.player2
    else:
        return gameState.player1


while True:
    line = sys.stdin.readline().strip()
    json_data = json.loads(line)

    gameState = GameState(json_data)

    player = myPlayer(gameState)
    enemy = enemyPlayer(gameState)

    if enemy.name == "Topic Team":
        position = enemy.position
        if(gameState.turn >= 200):
            if (enemy.position == [0,9] or enemy.position ==[9,0]) and player.xp >= 200:
                if enemy.daze_turns <= 1:
                    if(ValidateDaze(gameState)):
                        move_sequence.insert(0, "shop daze")
        if not move_sequence :
            move_sequence = player.GetDestroyFactorySequence(gameState)

    ores = count_letters_in_matrix(gameState.board)
    
    if not move_sequence and (enemy.processed_diamonds != 0 or enemy.processed_minerals != 0):
        move_sequence = player.GetDestroyFactorySequence(gameState)
    elif ores[0] > ores[1] * 2:
        if not move_sequence:
            if cnt_money % 3 < 2:
                move_sequence = player.GetMiningMoneySequence(gameState, 'D')
                cnt_money += 1
            else:
                move_sequence = player.GetMiningMoneySequence(gameState, 'M')
                cnt_money += 1

    elif ores[1] > 8:
        if not move_sequence:
            if cnt_xp % 5 < 4:
                move_sequence = player.GetMiningSequence(gameState, 'M')
                cnt_money += 1
    else: 
        if not move_sequence:
            move_sequence = player.GetMiningSequence(gameState, 'M')

    if(enemy.name != "Topic Team"):
        if (enemy.position == [0,9] or enemy.position ==[9,0]) and player.xp >= 20 and daze_cnt == 0:
                if enemy.daze_turns <= 1:
                    if(ValidateDaze(gameState)):
                        move_sequence.insert(0, "shop daze")
                    if(ValidateDaze(gameState)):
                        move_sequence.insert(0, "shop daze")
                        daze_cnt += 1000

    temp = move_sequence.pop(0)
    if temp.startswith('move') and player.daze_turns != 0:
        parts= temp.split(' ')
        targetX= int(parts[1])
        targetY= int(parts[2])
        dX= targetX - int(player.position[0])
        dY= targetY - int(player.position[1])
        newX= int(player.position[0]) - dX
        newY= int(player.position[1]) - dY
        temp=f'move {newX} {newY}'


    if temp.startswith('move'):
        px, py = player.position
        parts= temp.split(' ')
        tx = int(parts[1])
        ty = int(parts[2])
        ex, ey = enemy.position

        if ((tx - px) * (ey - py) - (ty - py) * (ex - px) == 0 and min(px, tx) <= ex <= max(px, tx) and min(py, ty) <= ey <= max(py, ty)):
            move_sequence.insert(0, f"move {tx} {ty}")
            #move_sequence.insert(0, "rest")
            #temp = move_sequence.pop(0)
            print("rest", flush=True)
            continue

    if(temp.startswith('attack')):
        attack_parts = temp.split(" ")
        x = int(attack_parts[1])
        y = int(attack_parts[2])

        if is_enemy_nearby(x, y, enemy.position):
            move_sequence = myPlayer.GetMiningSequence(gameState, 'M')
            print("rest", flush = True)

    parts = temp.split() 

    if len(parts) >= 3: 
        command = parts[0]
        if(command == "mine"):
            x = int(parts[1])  
            y = int(parts[2])  

            ore = gameState.board[x][y]
            prts = ore.split('_')
            remaining = int(prts[1])
            if remaining == 0: #invalid komanda za mine
                move_sequence.clear()
                move_sequence = player.GetMiningSequence(gameState, "M")
                temp = move_sequence.pop(0)     

    print(temp, flush=True)
