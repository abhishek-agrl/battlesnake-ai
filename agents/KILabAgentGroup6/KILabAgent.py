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
from environment.Battlesnake.util.kl_priority_queue import KLPriorityQueue


class KILabAgent(BaseAgent):
    def get_name(self):
        return "NAME"

    def get_head(self):
        return "viper"

    def get_tail(self):
        return "bolt"

    def start(self, game_info: GameInfo, turn: int, board: BoardState, you: Snake):
        self.food_memory: dict[Position, int] = {}
        self.view_radius = game_info.ruleset_settings.viewRadius
        self.turn = turn
        pass

    def move(
        self, game_info: GameInfo, turn: int, board: BoardState, you: Snake
    ) -> MoveResult:
        grid_map: GridMap[Occupant] = board.generate_grid_map()

        self.update_food_memory(you, board, turn)

        food_action = self.follow_food(you, grid_map)
        if food_action is not None:
            return MoveResult(direction=food_action)

        random_action = self.random_action(you, board)

        return MoveResult(direction=random_action)

    def end(self, game_info: GameInfo, turn: int, board: BoardState, you: Snake):
        pass

    def manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def update_food_memory(self, snake: Snake, board: BoardState, current_turn: int):
        head = snake.get_head()

        keys = list(self.food_memory.keys())

        for pos in keys:
            distance = self.manhattan_distance(head, pos)
            # There is no possible new information about these food items
            # view_radius of 1 means that the snake can only see the adjacent 4 fields
            if distance > self.view_radius:
                continue

            # Check if the food in view range is still there
            if pos not in board.food:
                self.food_memory.pop(pos)
                # print(f"Removing food from memory: {pos}")

        for food in board.food:
            self.food_memory[food] = current_turn


    def follow_food(self, snake: Snake, grid_map: GridMap) -> Optional[Direction]:
        head = snake.get_head()

        closest_distance = math.inf
        action = None

        for food in self.food_memory.keys():
            result = KILabAgent.a_star_search(head, food, grid_map)

            if result is None:
                continue

            distance, path = result

            if distance < closest_distance:
                closest_distance = distance
                action = path[0][1]

        return action

    def random_action(self, snake: Snake, board: BoardState) -> Direction:
        # select a random action from the possible actions without hitting a wall

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


    @staticmethod
    def a_star_search(
        start_field: Position,
        search_field: Position,
        grid_map: GridMap,
    ) -> Optional[Tuple[int, List[Tuple[Position, Direction | None]]]]:
        queue = KLPriorityQueue()

        came_from: dict[Position, Tuple[Position, Direction]] = { }

        cost_so_far: dict[Position, int] = {
            start_field: 0
        }

        queue.put(0, start_field)

        def herusitik(field, goal):
            # euclidian distance
            return math.sqrt((field.x - goal.x) ** 2 + (field.y - goal.y) ** 2)

        while not queue.empty():
            current = queue.get()

            if current == search_field:
                break

            for dir in Direction:
                pos = current.advanced(dir)

                if not grid_map.is_valid_at(pos.x, pos.y):
                    continue

                if grid_map.get_value_at_position(pos) == Occupant.Snake:
                    continue

                new_cost = cost_so_far[current] + 1
                old_cost = cost_so_far.get(pos, None)

                if old_cost is None or new_cost < old_cost:
                    cost_so_far[pos] = new_cost
                    priority = new_cost + herusitik(pos, search_field)
                    queue.put(priority, pos)
                    came_from[pos] = (current, dir)

        if search_field not in cost_so_far:
            # Could not find a path
            # print(f"Could not find a path from {start_field} to {search_field}")
            return None

        # Berechnung des Pfades
        cost = cost_so_far[search_field]
        path: List[Tuple[Position, Direction | None]] = []

        current = search_field
        last_action = None

        while current != start_field:
            path.append((current, last_action))
            child = came_from[current]

            last_action = child[1]
            current = child[0]

        path.append((start_field, last_action))

        path.reverse()

        return cost, path
