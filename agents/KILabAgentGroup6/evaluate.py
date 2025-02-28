from typing import List, Optional, Tuple
from Battlesnake.model.Direction import Direction
from Battlesnake.model.Occupant import Occupant
from Battlesnake.model.Snake import Snake
from Battlesnake.model.board_state import BoardState
from Battlesnake.model.Position import Position
from Battlesnake.model.grid_map import GridMap
from Battlesnake.helper.DirectionUtil import DirectionUtil
import numpy.typing as npt
import math
import numpy as np
import copy
import time

from Battlesnake.util.kl_priority_queue import KLPriorityQueue


UNKNOWN_BODY = Position(-1, -1)
DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

def manhattan_distance(pos1: Position, pos2: Position) -> int:
    return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

def get_snake_size(snake: Snake):
    if get_head(snake) == Position(-1, -1):
        return -1
    if snake.get_tail() == Position(-1, -1):
        return -1
    if Position(-1, -1) in snake.body:
        return -1
    return snake.get_length()

def flood_fill_score(start: Position, radius: int, grid_map: GridMap) -> float:
    if radius == -1:
        radius = min(grid_map.width, grid_map.height)

    total = (radius * radius) + ((radius + 1) * (radius + 1))
    found = 0
    visited: set[Position] = set()

    frontier: list[Position] = [start]

    while len(frontier) > 0:
        current = frontier.pop()

        for dir in DIRECTIONS:

            pos = current.advanced(dir)

            if manhattan_distance(start, pos) > radius:
                continue

            if pos in visited:
                continue

            visited.add(pos)

            if not grid_map.is_valid_at(pos.x, pos.y):
                continue

            value = grid_map.get_value_at_position(pos)

            if value is Occupant.Snake:
                continue

            found += 1

            frontier.append(pos)

    return found / total

def do_flood_fill(snake: Snake, grid: GridMap, radius: int):
    curr_pos = get_head(snake)
    if not grid.is_valid_at(x=curr_pos.x, y=curr_pos.y):
        return None
    
    dir_dict, bad_dir_grids = {}, []

    ## Check for each direction, and see if the snake would be safe
    for dir in DirectionUtil.possible_actions(snake.get_current_direction()):
        new_grid = copy.deepcopy(grid) # init temp grid  
        dir_dict[dir] = 0
        new_grid.set_value_at_position(curr_pos, -1)
        start_pos = get_next_pos(curr_pos, dir)

        frontier = []
        if check_empty_pos(new_grid, start_pos):
            new_grid.set_value_at_position(start_pos, -1)
            frontier.append(start_pos) 
            dir_dict[dir] += 1
        else:
            continue

        is_dir_safe = False
        while len(frontier) > 0:
            ff_pos: Position = frontier.pop(0)

            if manhattan_distance(ff_pos, curr_pos) > radius:
                continue
            
            for neighbour in get_neighbours(ff_pos):
                if check_empty_pos(new_grid, neighbour):
                    new_grid.set_value_at_position(neighbour, -1)
                    frontier.append(neighbour)
                    dir_dict[dir] += 1

            if snake.get_length() < dir_dict[dir]:
                is_dir_safe = True
                break

        if not is_dir_safe:
            bad_dir_grids.append(new_grid)

    sorted_dir_tup = sorted(dir_dict.items(), key=lambda item: item[1])
    allowed_actions = []
    for dir, available_space in sorted_dir_tup:
        if available_space >= snake.get_length():
            allowed_actions.append(dir)
    
    if len(allowed_actions) == 0:
        allowed_actions.append(sorted_dir_tup[0][0])
    
    return allowed_actions, bad_dir_grids

def get_head(snake: Snake):
    if not snake.is_alive() or len(snake.body) <= 1:
        return Position(-1,-1)
    else:
        return snake.get_head()
    
def check_empty_pos(grid:GridMap, pos: Position):
    value_at_pos = grid.get_value_at_position(pos)

    return not (value_at_pos == Occupant.Snake or value_at_pos == -1 or value_at_pos is None)

def get_neighbours(pos: Position):
    return [
        Position(pos.x+1, pos.y),
        Position(pos.x, pos.y+1),
        Position(pos.x-1, pos.y),
        Position(pos.x, pos.y-1),
    ]

def get_next_pos(position:Position, direction: Direction):
    if direction.name == "UP":
        return Position(position.x, position.y + 1)
    elif direction.name == "RIGHT":
        return Position(position.x + 1, position.y)
    elif direction.name == "DOWN":
        return Position(position.x, position.y - 1)
    elif direction.name == "LEFT":
        return Position(position.x - 1, position.y)
    else:
        return position

def a_star_search_eval(
    start_field: Position,
    search_field: Position,
    grid_map: GridMap,
    eval_grid: GridMap = None
) -> Optional[Tuple[int, List[Tuple[Position, Direction | None]]]]:
    queue = KLPriorityQueue()

    came_from: dict[Position, Tuple[Position, Direction]] = {}

    cost_so_far: dict[Position, int] = {start_field: 0}

    queue.put(0, start_field)

    def herusitik(field, goal):
        # euclidian distance
        return math.sqrt((field.x - goal.x) ** 2 + (field.y - goal.y) ** 2)

    start_time = time.time()
    while not queue.empty():
        current = queue.get()
        
        if time.time() - start_time > 2:
            print("A Star Failed")
            break

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

            if eval_grid is None:
                new_cost = cost_so_far[current] + 1
            else:
                new_cost = cost_so_far[current] + eval_grid[pos.x, pos.y]

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

def get_base_grid(np_grid_map: npt.NDArray, grid_center_val: int):
    base_grid = np.zeros_like(np_grid_map)
    radius = int(base_grid.shape[0] / 2)
    for i in reversed(range(1, radius + 2)):
        value = (grid_center_val - int(grid_center_val / radius * (radius - i + 1)))
        base_grid[:i, :] = value
        base_grid[:, :i] = value
        base_grid[-i:, :] = value
        base_grid[:, -i:] = value
    
    return base_grid

def get_border_grid(np_grid_map: npt.NDArray, border_val: int):
    border_grid = np.zeros_like(np_grid_map)
    border_grid[:1, :] = border_val
    border_grid[:, :1] = border_val
    border_grid[-1:, :] = border_val
    border_grid[:, -1:] = border_val
    
    return border_grid

def get_food_grid(np_grid_map: npt.NDArray, food_grid_memory: List, snake: Snake):
    # Constants
    food_grid_memory_size = 5
    
    current_food_grid = np.zeros_like(np_grid_map)
    food_idx = np.column_stack(np.where(np_grid_map == Occupant.Food))
    current_food_grid[food_idx[:, 0], food_idx[:, 1]] = 100
    food_grid_memory.insert(0,current_food_grid)
    
    if len(food_grid_memory)>food_grid_memory_size:
        food_grid_memory = food_grid_memory[:food_grid_memory_size+1]
    
    food_grid = np.zeros_like(np_grid_map)
    for i in range(len(food_grid_memory)):
        # Remove current viewable foods from older views
        if i != 0:
            food_grid_memory[i][food_idx[:,0],food_idx[:,1]]=0
        # Add all previous food memory with a discount
        try:
            food_grid += food_grid_memory[i] / (i+1)
        except:
            food_grid_memory = []

    health_score = math.pow( 10 - snake.health/10, 2)
    food_grid *= health_score

    return food_grid, food_grid_memory

def get_tails_grid(np_grid_map: npt.NDArray, board: BoardState, grid_map: GridMap, you: Snake):
    
    tails_grid= np.zeros_like(np_grid_map)

    for snake in board.snakes:
        if len(snake.body) < 3 or not snake.is_alive():
            continue

        res = a_star_search_eval(get_head(you), snake.get_tail(), grid_map)
        if res == None:
            return tails_grid
        
        tail_dist, _ = res
        if tail_dist > 3:
            continue

        tail_dir= DirectionUtil.direction_to_reach_field(snake.body[-2], snake.body[-1])
        possible_dirs = DirectionUtil.possible_actions(tail_dir)
        tail = snake.get_tail()

        if possible_dirs is None:
            continue

        for dir in possible_dirs:
            next_pos = tail.advanced(dir)
            
            if board.is_out_of_bounds(next_pos):
                continue

            tails_grid[next_pos.x, next_pos.y] = 100
    
    return tails_grid

def get_enemy_head_grid(np_grid_map: npt.NDArray, board: BoardState, you: Snake, enemy_head_val: int):
    enemy_head_grid = np.zeros_like(np_grid_map)
    head = get_head(you)

    for snake in board.snakes:
        snake_head = get_head(snake)
        if snake_head.is_position_equal_to(head):
            continue
        if manhattan_distance(head, snake_head) > enemy_head_val:
            continue
        enemy_head_grid += get_immediate_grid(np_grid_map, enemy_head_val, snake, 5)
    
    return enemy_head_grid
    
def get_immediate_grid(np_grid_map: npt.NDArray, enemy_head_val: int,  snake: Snake, weight:int = 5): 
    index_matrix = np.stack(np.indices(np_grid_map.shape), axis=-1)
    head = get_head(snake)

    diff = np.abs(index_matrix - index_matrix[head.x, head.y])
    manhattan_distances = np.sum(diff, axis=-1)
    manhattan_distances[manhattan_distances > enemy_head_val] = enemy_head_val
    manhattan_distances = weight - manhattan_distances
    return manhattan_distances * weight

def get_floodfill_grid(np_grid_map: npt.NDArray, bad_dir_grids: List[GridMap], board: BoardState, flood_fill_invalid_val: int):
    flood_fill_grid = np.zeros_like(np_grid_map)

    if bad_dir_grids is None:
        return flood_fill_grid
    
    for bad_grid in bad_dir_grids:
        np_bad_grid = np.array(bad_grid.grid_cache)
        for index in np.argwhere(np_bad_grid == -1):
            flood_fill_grid[index[0], index[1]] = flood_fill_invalid_val
            for neighbour in get_neighbours(Position(index[0], index[1])):
                if not board.is_out_of_bounds(neighbour):
                    flood_fill_grid[neighbour.x, neighbour.y] = flood_fill_invalid_val // 2
    
    return flood_fill_grid
            
def get_h2h_grid(np_grid_map: npt.NDArray, board: BoardState, you: Snake):
    h2h_grid = np.zeros_like(np_grid_map)
    head = get_head(you)

    for snake in board.snakes:
        snake_pos = get_head(snake)
        if snake_pos.is_position_equal_to(head):
            continue
        if snake_pos.is_position_equal_to(Position(-1,-1)) or manhattan_distance(head, snake_pos) > 2:
            continue
        
        for dir in snake.possible_actions():
            snake_next_pos = snake_pos.advanced(dir)
            if board.is_out_of_bounds(snake_next_pos):
                continue
            
            # h2h_grid[snake_next_pos.x, snake_next_pos.y] = 100000
            snake_length = get_snake_size(snake)
            if snake_length != -1 and snake_length < you.get_length():
                h2h_grid[snake_next_pos.x, snake_next_pos.y] = 0
            else:
                h2h_grid[snake_next_pos.x, snake_next_pos.y] = 100000
    
    return h2h_grid
                            
def get_eval_grid( 
        flood_fill_invalid_val: int, 
        grid_center_val: int,
        you_head_val: int,
        enemy_head_val: int,
        border_val: int,
        enemy_snake_val: int,
        grid_map: GridMap,
        board: BoardState,
        food_grid_memory,
        bad_dirs_grid,
        you: Snake,
    ):
        
        np_grid_map = np.array(grid_map.grid_cache)

        ## All Positives
        base_grid = get_base_grid(np_grid_map, grid_center_val)
        # print("Base grid Done")
        my_head_grid = get_immediate_grid(np_grid_map, enemy_head_val, you)
        # print("My head grid done")
        food_grid, food_grid_memory = get_food_grid(np_grid_map, food_grid_memory, you)
        # print("Food grid done")
        tails_grid = get_tails_grid(np_grid_map, board, grid_map, you)
        # print("Tails grid done")
        
        ## Normalize the positive eval grid
        positive_grid = base_grid + my_head_grid + tails_grid + food_grid
        # print("Positive grid done")
        max_grid_val = min(1, positive_grid.max())
        # print("max_grid_val done")
        positive_grid = positive_grid / max_grid_val * 100
        # print("positive grid normalization done")
        
        ## Invert the grid (because A* minimizes the cost)
        max_val_grid = np_grid_map.copy()
        # print("max_val_grid done")
        max_val_grid[:, :] = positive_grid.max()
        # print("max_val_grid copy done")
        positive_grid = max_val_grid - positive_grid
        # print("positive grid max_val diff done")

        ## All Negatives (Adds to the cost)
        border_grid = get_border_grid(np_grid_map, border_val)
        # print("border grid done")
        flood_fill_grid = get_floodfill_grid(np_grid_map, bad_dirs_grid, board, flood_fill_invalid_val)
        # print("flood_fill grid done")
        h2h_collision_grid = get_h2h_grid(np_grid_map, board, you)
        # print("h2h collision grid done")
        enemy_head_grid = get_enemy_head_grid(np_grid_map, board, you, enemy_head_val)
        # print("enemy head grid done")

        final_grid = positive_grid + border_grid + flood_fill_grid + h2h_collision_grid + enemy_head_grid
        # print("final grid done")

        final_grid = final_grid.astype(int)

        snake_idx = np.column_stack(np.where(np_grid_map == Occupant.Snake))
        # print("snake_idx done")

        final_grid[snake_idx[:, 0], snake_idx[:, 1]] = enemy_snake_val
        # print("final_grid enemy_snake done")
        final_grid[you.get_head().x, you.get_head().y] = you_head_val
        
        # Replacing all Negatives with zero to avoid loop in A*
        final_grid[final_grid<0] = 0
        # print("final_grid you_head_val done")

        return final_grid, food_grid_memory
