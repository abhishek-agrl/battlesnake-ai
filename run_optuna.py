import time
from agents.KILabAgentGroup6.KILabAgent import KILabAgent
from agents.KILabAgentGroup6.KILabAgentGridEval import KILabAgentGridEval
from environment.Battlesnake.model.RulesetSettings import RulesetSettings
from environment.Battlesnake.modes.Modes import GameMode
from environment.battlesnake_environment import BattlesnakeEnvironment
from environment.Battlesnake.agents.RemoteAgent import RemoteAgent
from agents.RandomAgent.RandomAgent import RandomAgent
import gc
import optuna

def create_env(agents): 
    return BattlesnakeEnvironment(
        width=11,
        height=11,
        agents=agents,
        act_timeout=0.5,
        export_games=False,
        mode=GameMode.STANDARD,
        squad_assignments=[1, 1, 3, 3, 4, 4],
        ruleset_settings=RulesetSettings(viewRadius=-1),
        dark_mode=True,
        multiprocessing_step=False,
        speed_initial=2,   
    )

def play_game(
        ff_invalid_val: int, 
        border_val: int, 
        enemy_head_val: int, 
        enemy_snake_val: int, 
        grid_center_val: int, 
        you_head_val: int, 
        num_games=20):
    
    agent_grid_eval = KILabAgentGridEval(
        flood_fill_invalid_val=ff_invalid_val,
        border_val=border_val,
        enemy_head_val=enemy_head_val,
        enemy_snake_val=enemy_snake_val,
        grid_center_val=grid_center_val,
        you_head_val=you_head_val
    )
    agent_a_star = KILabAgent()
    # agent_c = RemoteAgent("130.75.31.196:8019")
    # agent2 = KILabAgent()

    # agents = [agent1, agent2]
    agents = [agent_grid_eval, agent_a_star]
    env = create_env(agents=agents)
    env.reset()
    env.render()

    score = 0
    count = 0
    my_turns = 0
    all_turns = 0
    while True:
        step_start_time = time.time()
        env.handle_input()

        end = env.step()
        all_turns += 1
        if not env.board.get_dead_snake_by_id(env.snake_ids[0]):
            my_turns += 1

        #if all(agent_dead) or end:
        if end:
            count+=1
            turn_ratio = my_turns/all_turns
            if env.board.get_dead_snake_by_id(env.snake_ids[0]):
                score += turn_ratio-1.1
            else:
                score += turn_ratio
            print(f"Game {count}, Running Score: {score}")
            if count > num_games:
                break
            env.reset()
            env.cleanup()
            env = create_env(agents=agents)
            env.reset()
            my_turns = 0
            all_turns = 0

        env.render()
        step_time = int((time.time() - step_start_time) * 1000)

        env.wait_after_step(step_time)
    
    return score

def objective(trial: optuna.Trial):
    flood_fill_invalid_val = trial.suggest_int('flood_fill_invalid_val', 1, 1000, step=5)
    border_val = trial.suggest_int('border_val', 1, 1000, step=5)
    enemy_head_val = trial.suggest_int('enemy_head_val', 1, 1000, step=5)
    enemy_snake_val = trial.suggest_int('enemy_snake_val', 1, 1000, step=5)
    grid_center_val = trial.suggest_int('grid_center_val', 1, 1000, step=5)
    you_head_val = trial.suggest_int('you_head_val', 1, 1000, step=5)

    score = play_game(
        ff_invalid_val=flood_fill_invalid_val, 
        border_val=border_val, 
        enemy_head_val=enemy_head_val, 
        enemy_snake_val=enemy_snake_val, 
        grid_center_val=grid_center_val, 
        you_head_val=you_head_val,
        num_games=10)
    print("##############################################")
    print("##############################################")
    print(f'Score: {score} -> For weights: flood_fill_invalid_val: {flood_fill_invalid_val}, border_val: {border_val}, enemy_head_val: {enemy_head_val}, enemy_snake_val:{enemy_snake_val}, grid_center_val: {grid_center_val}, you_head_val: {you_head_val}')
    print("##############################################")
    print("##############################################")

    gc.collect()
    return score

    


if __name__ == "__main__":

    # play_game(weights= [5, 8, 3, 4, 6, 7], num_games=10)

    # Optuna Learning Block
    study = optuna.create_study(study_name="Snake_learn", storage="sqlite:///db.sqlite3", direction="maximize", sampler=optuna.samplers.TPESampler(seed=17), load_if_exists=True)
    study.optimize(objective, n_trials=100)
    print(study.best_params)

    # # health_weight, length_weight, fd_weight, fr_weight, h2h_weight, flood_weight
    # agent1 = KILabAgentGridEval()
    # # agent_base = RemoteAgent("130.75.31.196:8015")
    # agent_a = KILabAgentNash()
    # # agent_a = RemoteAgent("130.75.31.196:8018")
    # # agent_b = RemoteAgent("130.75.31.196:8018")
    # # agent_c = RemoteAgent("130.75.31.196:8019")
    # agents = [agent1, agent_a]
    # # remote_agent = RemoteAgent(url='130.75.31.196:8000')
    # # agents.append(remote_agent)

    # env = BattlesnakeEnvironment(
    #     width=11,
    #     height=11,
    #     agents=agents,
    #     act_timeout=1,
    #     export_games=True,
    #     mode=GameMode.STANDARD,
    #     squad_assignments=[1, 1, 3, 3, 4, 4],
    #     ruleset_settings=RulesetSettings(viewRadius=-1),
    #     dark_mode=True,
    #     multiprocessing_step=False,
    #     speed_initial=3,
    # )

    # env.reset()
    # env.render()

    # lost, won = 0, 0
    # count = 0
    # while True:
    #     step_start_time = time.time()
    #     env.handle_input()
    #     end = env.step()

    #     if end:
    #         count+=1
    #         if env.board.get_dead_snake_by_id(env.snake_ids[0]):
    #             lost += 1
    #         else:
    #             won += 1
    #         print(f"Won: {won}, Lost: {lost}, Percentage: {won/(won+lost)*100}")
    #         if count > 100:
    #             break
    #         env.reset()
    #         env.cleanup()
    #         env = create_env(agents=agents)
    #         env.reset()
            

    #     env.render()
    #     step_time = int((time.time() - step_start_time) * 1000)

    #     env.wait_after_step(step_time)
    
    # print(f"Won: {won}, Lost: {lost}, Percentage: {won/(won+lost)*100}")