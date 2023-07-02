from utils import *

def analyse_cell(env, og_cell):
    front_cell = env.agent_pos + env.agent_dir
    front_type = env.grid.get(front_cell[0], front_cell[1])
    
    pass