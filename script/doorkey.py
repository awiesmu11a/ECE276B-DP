from utils import *
from example import example_use_of_gym_env

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def doorkey_problem(env):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """

    agent_dir = env.dir_vec
    agent_pos = env.agent_pos
    
    dir_dict = {(0,-1):[0,0],
                (1,0):[0,1],
                (-1,0):[1,0],
                (0,1):[1,1]}
    
    control_space = np.array([[1, 0, 0, 0, 0],
                              [0, 0, 0, -1, -1],
                              [0, 0, 0, 1, -1],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0]])


    state = np.array([agent_pos[0],
                    agent_pos[1], 
                    agent_dir[0], 
                    agent_dir[1], 
                    0, 
                    0])
    
    start_state = state.copy()
    
    goal_reached = False
    timestep = 0
    
    costs_to_go = np.ones((env.width, env.height, 2, 2, 2)) * 1000
    actions = np.zeros((env.width, env.height))
    
    dir_indices = dir_dict[(state[2], state[3])]

    costs_to_go[state[0], state[1]][dir_indices[0], dir_indices[1]][state[4]] = 0
    
    optim_act_seq = []
    
    open_list = []


    front_cell = env.front_pos
    element = env.grid.get(front_cell[0], front_cell[1])
    
    if element is None:
        open_list.append((state, control_space[0]))
        open_list.append((state, control_space[1]))
        open_list.append((state, control_space[2]))
    
    elif element.type == 'goal':
        goal_reached = True
        print("Goal Reached!")
        return optim_act_seq
    
    elif element.type == 'door':
        if element.is_open:
            open_list.append((state, control_space[0]))
            open_list.append((state, control_space[1]))
            open_list.append((state, control_space[2]))
        else:
            open_list.append((state, control_space[3]))
            open_list.append((state, control_space[4]))
    
    elif element.type == 'key':
        open_list.append((state, control_space[3]))
        open_list.append((state, control_space[0]))
        open_list.append((state, control_space[1]))
        open_list.append((state, control_space[2]))
    
    elif element.type == 'wall':
        open_list.append((state, control_space[1]))
        open_list.append((state, control_space[2]))


    while len(open_list) > 0:

        current_state, current_action = open_list.pop(0)

        B = B_matrix(current_state)

        x_prime = current_state + B @ (current_action.T)

        """    
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Current State: ", current_state)
        print("Current Action: ", current_action)
        print("Next State: ", x_prime)
        """
    

        if (current_action == control_space[4]).all():
            _, done = step(env, UD)
        


        dir_indices_update = dir_dict[(x_prime[2], x_prime[3])]
        dir_indices = dir_dict[(current_state[2], current_state[3])]

        costs_to_go[x_prime[0], x_prime[1]][dir_indices_update[0], dir_indices_update[1]][x_prime[4]] = min(1 + costs_to_go[current_state[0], current_state[1]][dir_indices[0], dir_indices[1]][current_state[4]],
                                                                                                            costs_to_go[x_prime[0], x_prime[1]][dir_indices_update[0], dir_indices_update[1]][x_prime[4]])

        current_front_cell = (x_prime[0] + x_prime[2], x_prime[1] + x_prime[3])
        element = env.grid.get(current_front_cell[0], current_front_cell[1])

        if not (costs_to_go[x_prime[0], x_prime[1]][dir_indices_update[0], dir_indices_update[1]][x_prime[4]] == (1 + costs_to_go[current_state[0], current_state[1]][dir_indices[0], dir_indices[1]][current_state[4]])):
            if not x_prime[5]:
                continue

        if element is None:
            open_list.append((x_prime, control_space[0]))
            open_list.append((x_prime, control_space[1]))
            open_list.append((x_prime, control_space[2]))
            continue

        if element.type == "goal":
            optim_act_seq.append(MF)
            goal_reached = True
            state = x_prime
            break
        
        if element.type == "key":
            if not x_prime[4]:
                open_list.append((x_prime, control_space[3]))
        
        if element.type == "door":
            if x_prime[5] == 1:
                open_list.append((x_prime, control_space[0]))
            elif x_prime[4] == 1:
                open_list.append((x_prime, control_space[4]))
        
        open_list.append((x_prime, control_space[1]))
        open_list.append((x_prime, control_space[2]))
    
    print(costs_to_go[0,0].shape)
    
    for i in range(env.width):
        print("-----------------------------------------------")
        for j in range(env.height):
            if np.min(costs_to_go[j,i]) == 1000:
                print("X", end="|")
            else:
                print(np.min(costs_to_go[j, i]), end="||")
                actions[j, i] = np.argmin(costs_to_go[j, i])
        print()
    
    print("Actions:")

    for i in range(env.width):
        print("-----------------------------------------------")
        for j in range(env.height):
            print(actions[j, i], end="|")
        print()
    
    # Finding the optimal action sequence



    return optim_act_seq


def partA():

    env_path = "./envs/known_envs/doorkey-8x8-shortcut.env"
    print("Environment Name: ", env_path.split("/")[-1].split(".")[0])
    env, info = load_env(env_path)  # load an environment
    #plot_env(env)
    seq = doorkey_problem(env)  # find the optimal action sequence
    print(seq)
    #draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save


def partB():
    env_folder = "./envs/random_envs"
    env, info, env_path = load_random_env(env_folder)
    seq = doorkey_problem(env)  # find the optimal action sequence
    print(seq)
    #draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save


def B_matrix(x):
    B = np.array([[x[2], 0, 0, 0, 0],
                  [x[3], 0, 0, 0, 0],
                  [0, 0, 0, x[3], x[2]],
                  [0, 0, 0, x[2], x[3]],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0]])
    return B


if __name__ == "__main__":
    # example_use_of_gym_env()
    partA()
    # partB()
