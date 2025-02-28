
from agents.KILabAgentGroup6.KILabAgentGridEval import KILabAgentGridEval
from environment.Battlesnake.server import BattlesnakeServer

agent = KILabAgentGridEval()  # TODO select your agent
port = 8005  # TODO set your port

BattlesnakeServer.start_server(agent=agent, port=port)
