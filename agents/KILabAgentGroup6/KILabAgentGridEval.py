import math
from typing import List, Optional, Tuple

import numpy as np

from environment.Battlesnake.agents.BaseAgent import BaseAgent
from environment.Battlesnake.helper.DirectionUtil import DirectionUtil
from environment.Battlesnake.model.board_state import BoardState
from environment.Battlesnake.model.Direction import Direction
from environment.Battlesnake.model.GameInfo import GameInfo
from environment.Battlesnake.model.grid_map import GridMap
from environment.Battlesnake.model.MoveResult import MoveResult
from environment.Battlesnake.model.Occupant import Occupant
from environment.Battlesnake.model.Position import Position
from environment.Battlesnake.model.Snake import Snake

from agents.KILabAgentGroup6.evaluate import get_head, do_flood_fill, get_eval_grid, a_star_search_eval

import time


class KILabAgentGridEval(BaseAgent):

    def __init__(
            self, 
            flood_fill_invalid_val = 596, 
            grid_center_val = 116, 
            enemy_snake_val = 641, 
            you_head_val = 241,
            enemy_head_val = 711, 
            border_val = 271
            ):
        super().__init__()
        

        self.flood_fill_invalid_val = flood_fill_invalid_val
        self.grid_center_val = grid_center_val
        self.enemy_snake_val = enemy_snake_val
        self.you_head_val = you_head_val
        self.enemy_head_val = enemy_head_val
        self.border_val = border_val
        
        self.view_radius_by_id = {}
        self.food_memory_by_id = {}
        self.food_grid_memory_by_id = {}
        self.bad_dirs_grid_by_id = {}
        self.flood_fill_allowed_actions_by_id = {}
        self.turn_by_id = {}
        self.board_by_id = {}
        self.grid_map_by_id = {}

    def get_name(self):
        return "NAME_GRID_EVAL"

    def get_head(self):
        return "viper"

    def get_tail(self):
        return "bolt"

    def start(self, game_info: GameInfo, turn: int, board: BoardState, you: Snake):
        game_id = game_info.id ## All Variables should be a dict that refer the same game
        
        food_memory: dict[Position, int] = {}
        
        self.food_memory_by_id[game_id] = food_memory
        self.food_grid_memory_by_id[game_id] = []
        self.view_radius_by_id[game_id] = game_info.ruleset_settings.viewRadius
        self.turn_by_id[game_id] = turn

        self.board_by_id[game_id] = board
        self.grid_map_by_id[game_id] = board.generate_grid_map()
        
        self.flood_fill_allowed_actions_by_id[game_id] = None
        self.bad_dirs_grid_by_id[game_id] = None


    def move(
        self, game_info: GameInfo, turn: int, board: BoardState, you: Snake
    ) -> MoveResult:
        save_time = time.time()
        game_id = game_info.id

        self.board_by_id[game_id] = board
        self.update_food_memory(you, board, turn, game_id)
        
        grid_map: GridMap[Occupant] = board.generate_grid_map()
        self.grid_map_by_id[game_id] = grid_map
        # print("Generate Grid Map")

        ## Flood Fill Check
        res = do_flood_fill(you, grid_map, self.view_radius_by_id[game_id])
        if res is not None:
            self.flood_fill_allowed_actions_by_id[game_id], self.bad_dirs_grid_by_id[game_id] = res
        # print("Flood Fill Check")

        ## Generate the evaluated grid
        eval_grid, self.food_grid_memory_by_id[game_id] = get_eval_grid(
            border_val = self.border_val,
            enemy_head_val = self.enemy_head_val,
            bad_dirs_grid = self.bad_dirs_grid_by_id[game_id],
            board = board,
            enemy_snake_val = self.enemy_snake_val,
            flood_fill_invalid_val = self.flood_fill_invalid_val,
            food_grid_memory = self.food_grid_memory_by_id[game_id],
            grid_center_val = self.grid_center_val,
            grid_map = grid_map,
            you=you,
            you_head_val=self.you_head_val)
        # print("Eval Grid Done")
        min_cost = math.inf
        best_path = None
        next_direction = None

        ## Sort the eval_grid and get the lowest indices -> Even for 2d arrays it returns the linear index
        ## Then get the 2d grid location by converting the linear index to 2d index
        ## Then stack them on a 3rd dimesion to essentially make a list of 2d indices.
        sorted_indices = np.dstack(np.unravel_index(np.argsort(eval_grid, axis=None), eval_grid.shape)).tolist()[0]
        # print("Index Sorted")
        if sorted_indices is not None:
            # sorted_indices = sorted_indices[:100] if len(sorted_indices)>100 else sorted_indices
            # print("Index Sort Select")
            for index in sorted_indices:
                if time.time() - save_time > 0.25:
                    break

                if len(index) != 2:
                    continue

                x, y = index[0], index[1]
                min_value = max(0.1, eval_grid[x,y])
                if get_head(you).is_position_equal_to(Position(x,y)) or Position(x,y) in self.get_invalid_positions(game_id):
                    continue
                
                # print("A Star Eval grid start")
                res = a_star_search_eval(get_head(you), Position(x,y), grid_map, eval_grid)
                # print("A Star Eval grid finish")
                if res is None:
                    continue
                
                search_cost, search_path = res

                if min_value*search_cost < min_cost and search_path is not None and len(search_path) > 0:
                    min_cost = min_value*search_cost
                    best_path = search_path

            if best_path is None or len(best_path) == 0:
                next_direction = self.follow_food(you, game_id)
                # print("Follow Food")
            else:
                next_direction = best_path[0][1]
                # print("Follow Best Path")
                
        else:
            next_direction = self.follow_food(you, game_id)
            # print("Follow Food Sorted Indices None")

        if next_direction is not None:
            return MoveResult(next_direction)
        
        # print("Random Action")
        return MoveResult(self.random_action(you, board))
            

    def end(self, game_info: GameInfo, turn: int, board: BoardState, you: Snake):
        game_id = game_info.id
        self.view_radius_by_id.pop(game_id, None)
        self.food_memory_by_id.pop(game_id, None)
        self.food_grid_memory_by_id.pop(game_id, None)
        self.bad_dirs_grid_by_id.pop(game_id, None)
        self.flood_fill_allowed_actions_by_id.pop(game_id, None)
        self.turn_by_id.pop(game_id, None)
        self.board_by_id.pop(game_id, None)
        self.grid_map_by_id.pop(game_id, None)
        pass

    def manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def get_invalid_positions(self, game_id):
        invalid_positions = []
        for snake in self.board_by_id[game_id].snakes:
            invalid_positions+=snake.body
        
        return list(set(invalid_positions))

    def update_food_memory(self, snake: Snake, board: BoardState, current_turn: int, game_id):
        head = snake.get_head()

        keys = list(self.food_memory_by_id[game_id].keys())

        for pos in keys:
            distance = self.manhattan_distance(head, pos)
            # There is no possible new information about these food items
            # view_radius of 1 means that the snake can only see the adjacent 4 fields
            if distance > self.view_radius_by_id[game_id]:
                continue

            # Check if the food in view range is still there
            if pos not in board.food:
                self.food_memory_by_id[game_id].pop(pos)
                # print(f"Removing food from memory: {pos}")

        for food in board.food:
            self.food_memory_by_id[game_id][food] = current_turn

    def follow_food(self, snake: Snake, game_id) -> Optional[Direction]:
        head = snake.get_head()

        closest_distance = math.inf
        action = None

        manhatten = {}

        for food in self.food_memory_by_id[game_id].keys():
            distance = self.manhattan_distance(head, food)
            manhatten[food] = distance

        # sort by distance
        sorted_food = dict(sorted(manhatten.items(), key=lambda item: item[1]))

        # take the 5 closest foods
        sorted_food = dict(list(sorted_food.items())[:5])

        for food in sorted_food.keys():
            result = a_star_search_eval(head, food, self.grid_map_by_id[game_id])

            if result is None:
                continue

            distance, path = result

            if distance < closest_distance:
                closest_distance = distance
                action = path[0][1]

        if action not in self.flood_fill_allowed_actions_by_id[game_id]:
            # print(f"not chasing food cause path is blocked")
            return None
        
        return action

    def random_action(self, snake: Snake, board: BoardState) -> Direction:
        # select a random action from the possible actions without hitting a wall
        ## TODO: Check for better random algorithm
        possible_actions = snake.possible_actions()
        head = snake.get_head()

        if head is None:
            return Direction.UP

        if possible_actions is None:
            return Direction.UP

        possible_actions_filtered = []
        for action in possible_actions:
            neighbor_position = head.advanced(action)

            if board.is_out_of_bounds(neighbor_position):
                continue

            if board.is_occupied_by_snake(neighbor_position):
                continue

            possible_actions_filtered.append(action)

        if len(possible_actions_filtered) == 0:
            return Direction.UP

        return np.random.choice(possible_actions_filtered)
