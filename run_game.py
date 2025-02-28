import time

from agents.KILabAgentGroup6.KILabAgent import KILabAgent
from agents.KILabAgentGroup6.KILabAgentGridEval import KILabAgentGridEval
from agents.RandomAgent.RandomAgent import RandomAgent
from environment.Battlesnake.model.RulesetSettings import RulesetSettings
from environment.Battlesnake.modes.Modes import GameMode
from environment.battlesnake_environment import BattlesnakeEnvironment
from environment.Battlesnake.agents.RemoteAgent import RemoteAgent

if __name__ == "__main__":
    agent1 = KILabAgentGridEval(
        border_val=766,
        enemy_head_val=6,
        enemy_snake_val=531,
        flood_fill_invalid_val=946,
        grid_center_val=106,
        you_head_val=131
    )
    agent2 = KILabAgent()
    agent3 = RandomAgent()
    agents = [agent1, agent2, agent3]
    # remote_agent = RemoteAgent(url='130.75.31.196:8000')
    # agents.append(remote_agent)

    env = BattlesnakeEnvironment(
        width=11,
        height=11,
        agents=agents,
        act_timeout=0.5,
        export_games=True,
        mode=GameMode.STANDARD,
        squad_assignments=[1, 1, 3, 3, 4, 4],
        ruleset_settings=RulesetSettings(viewRadius=-1),
        dark_mode=True,
        multiprocessing_step=False,
        speed_initial=2
    )

    env.reset()
    env.render()

    while True:
        step_start_time = time.time()
        env.handle_input()
        env.step()
        env.render()
        step_time = int((time.time() - step_start_time) * 1000)

        env.wait_after_step(step_time)
