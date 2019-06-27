import os
import numpy as np


from sweeper import Sweeper
from visualizer import RunLines, RunLinesIndividual


def parse_steps_log(log_path, max_steps, interval=0):
    """
    Retrieves num_steps at which episodes were completed
    and returns an array where the value at index i represents
    the number of episodes completed until step i
    """
    print(log_path)
    with open(log_path, "r") as f:
        lines = f.readlines()
    rewards_over_time = np.zeros(max_steps//interval)
    try:
        num_steps = get_max_steps(lines)
        if num_steps < max_steps:
            return None
        for line in lines:
            if 'total steps' not in line:
                continue
            num_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            if num_steps > max_steps:
                break
            reward = float(line.split("|")[1].split(",")[2].split("/")[0].split(" ")[-1])
            rewards_over_time[int(num_steps//interval)-1] = reward
        return rewards_over_time
    except:
        return None

def get_max_steps(lines):
    for line in lines[::-1]:
        if 'total steps' in line:
            max_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            return max_steps
    return -1


def draw_lunar_lander_dqn(settings, save_path):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps in 10000s", ylabel="Average Reward",
                 interval=10000)
    v.draw()


def parse_reject_ratio(log_path, max_steps, interval=0):
    """
    Retrieves num_steps at which episodes were completed
    and returns an array where the value at index i represents
    the number of episodes completed until step i
    """
    print(log_path)

    ratio_type = log_path.split("/")[-1]
    log_path = '/'.join(log_path.split("/")[:-1])


    with open(log_path, "r") as f:
        lines = f.readlines()
    ratio_over_time = np.zeros(max_steps//interval)
    try:
        num_steps = get_max_steps(lines)
        if num_steps < max_steps:
            return None
        for line in lines:
            if 'total steps' not in line:
                continue
            num_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            if num_steps > max_steps:
                break
            if ratio_type == 'oracle':
                ratio = float(line.split("|")[1].split(",")[3].split(" ")[-1])
            else:
                ratio = float(line.split("|")[1].split(",")[4].split(" ")[-1])
            ratio_over_time[int(num_steps//interval)-1] = ratio
        return ratio_over_time
    except:
        return None


def draw_reject_ratio(settings, save_path):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_reject_ratio
    path_formatters = []

    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, '/'.join(cf.split("/")[:-1])))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format + '/' + cf.split("/")[-1])

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps in 10000s", ylabel="Reject Ratio",
                 interval=10000, ylim=(0, 1.2))
    v.draw()



if __name__ == '__main__':

    # # One Room
    # settings = [("experiment/config_files/rooms/one_room/q.json", 0, 3, 500000, "Q"),
    #             ("experiment/config_files/rooms/one_room/q_e.json", 1, 3, 500000, "Q - E"),
    #             ("experiment/config_files/rooms/one_room/q_o.json", 2, 3, 500000, "Q - O"),
    #             ("experiment/config_files/rooms/one_room/q_t.json", 0, 3, 500000, "Q - T"),
    #             ("experiment/config_files/rooms/one_room/q_eo.json", 3, 3, 500000, "Q - EO"),
    #             ("experiment/config_files/rooms/one_room/q_te.json", 0, 3, 500000, "Q - TE"),
    #             ("experiment/config_files/rooms/one_room/q_to.json", 3, 3, 500000, "Q - TO"),
    #             ("experiment/config_files/rooms/one_room/q_teo.json", 3, 3, 500000, "Q - TEO")]
    #             # ("experiment/config_files/rooms/one_room/q_te2o.json", 2, 3, 500000, "Q - TE2O"),
    #             # ("experiment/config_files/rooms/one_room/q_te3o.json", 2, 3, 500000, "Q - TE3O"),
    #
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/rooms/one_room/plot_xy.pdf")
    # # #
    # # #
    # # #
    # # # # Two Rooms
    # settings = [("experiment/config_files/rooms/two_rooms/q.json", 0, 3, 500000, "Q"),
    #             ("experiment/config_files/rooms/two_rooms/q_e.json", 0, 3, 500000, "Q - E"),
    #             ("experiment/config_files/rooms/two_rooms/q_o.json", 2, 3, 500000, "Q - O"),
    #             ("experiment/config_files/rooms/two_rooms/q_t.json", 0, 3, 500000, "Q - T"),
    #             ("experiment/config_files/rooms/two_rooms/q_eo.json", 3, 3, 500000, "Q - EO"),
    #             ("experiment/config_files/rooms/two_rooms/q_te.json", 0, 3, 500000, "Q - TE"),
    #             ("experiment/config_files/rooms/two_rooms/q_to.json", 3, 3, 500000, "Q - TO"),
    #             ("experiment/config_files/rooms/two_rooms/q_teo.json", 2, 3, 500000, "Q - TEO")]
    #             # ("experiment/config_files/rooms/two_rooms/q_te2o.json", 2, 3, 500000, "Q - TE2O"),
    #             # ("experiment/config_files/rooms/two_rooms/q_te3o.json", 3, 3, 500000, "Q - TE3O"),
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/rooms/two_rooms/plot_xy.pdf")
    # # #
    # # #
    # # # # Hard Maze
    # settings = [("experiment/config_files/rooms/hard_maze/q.json", 2, 3, 500000, "Q"),
    #             ("experiment/config_files/rooms/hard_maze/q_e.json", 2, 3, 500000, "Q - E"),
    #             ("experiment/config_files/rooms/hard_maze/q_o.json", 2, 3, 500000, "Q - O"),
    #             ("experiment/config_files/rooms/hard_maze/q_t.json", 2, 3, 500000, "Q - T"),
    #             ("experiment/config_files/rooms/hard_maze/q_eo.json", 2, 3, 500000, "Q - EO"),
    #             ("experiment/config_files/rooms/hard_maze/q_te.json", 2, 3, 500000, "Q - TE"),
    #             ("experiment/config_files/rooms/hard_maze/q_to.json", 0, 3, 500000, "Q - TO"),
    #             ("experiment/config_files/rooms/hard_maze/q_teo.json", 2, 3, 500000, "Q - TEO")]
    #             # ("experiment/config_files/rooms/hard_maze/q_te2o.json", 1, 3, 500000, "Q - TE2O"),
    #             # ("experiment/config_files/rooms/hard_maze/q_te3o.json", 1, 3, 500000, "Q - TE3O"),
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/rooms/hard_maze/plot_xy.pdf")
    #
    #
    # # # Hard Maze
    # settings = [("experiment/config_files/laplace_agent_rooms/hard_maze/q.json", 0, 3, 500000, "Q"),
    #             ("experiment/config_files/laplace_agent_rooms/hard_maze/q_e.json", 0, 3, 500000, "Q - E"),
    #             ("experiment/config_files/laplace_agent_rooms/hard_maze/q_o.json", 3, 3, 500000, "Q - O"),
    #             ("experiment/config_files/laplace_agent_rooms/hard_maze/q_t.json", 0, 3, 500000, "Q - T"),
    #             ("experiment/config_files/laplace_agent_rooms/hard_maze/q_eo.json", 3, 3, 500000, "Q - EO"),
    #             ("experiment/config_files/laplace_agent_rooms/hard_maze/q_te.json", 0, 3, 500000, "Q - TE"),
    #             ("experiment/config_files/laplace_agent_rooms/hard_maze/q_to.json", 2, 3, 500000, "Q - TO"),
    #             ("experiment/config_files/laplace_agent_rooms/hard_maze/q_teo.json", 3, 3, 500000, "Q - TEO")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/laplace_agent_rooms/hard_maze/plot_onelayer.pdf")
    #
    #
    #
    #
    # # Two Rooms
    # settings = [("experiment/config_files/laplace_agent_rooms/two_rooms/q.json", 0, 3, 500000, "Q"),
    #             ("experiment/config_files/laplace_agent_rooms/two_rooms/q_e.json", 0, 3, 500000, "Q - E"),
    #             ("experiment/config_files/laplace_agent_rooms/two_rooms/q_o.json", 2, 3, 500000, "Q - O"),
    #             ("experiment/config_files/laplace_agent_rooms/two_rooms/q_t.json", 0, 3, 500000, "Q - T"),
    #             ("experiment/config_files/laplace_agent_rooms/two_rooms/q_eo.json", 2, 3, 500000, "Q - EO"),
    #             ("experiment/config_files/laplace_agent_rooms/two_rooms/q_te.json", 0, 3, 500000, "Q - TE"),
    #             ("experiment/config_files/laplace_agent_rooms/two_rooms/q_to.json", 2, 3, 500000, "Q - TO"),
    #             ("experiment/config_files/laplace_agent_rooms/two_rooms/q_teo.json", 1, 3, 500000, "Q - TEO")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/laplace_agent_rooms/two_rooms/plot_onelayer.pdf")
    #
    # # One ROOM
    # settings = [("experiment/config_files/laplace_agent_rooms/one_room/q.json", 0, 3, 500000, "Q"),
    #             ("experiment/config_files/laplace_agent_rooms/one_room/q_e.json", 0, 3, 500000, "Q - E"),
    #             ("experiment/config_files/laplace_agent_rooms/one_room/q_o.json", 3, 3, 500000, "Q - O"),
    #             ("experiment/config_files/laplace_agent_rooms/one_room/q_t.json", 0, 3, 500000, "Q - T"),
    #             ("experiment/config_files/laplace_agent_rooms/one_room/q_eo.json", 2, 3, 500000, "Q - EO"),
    #             ("experiment/config_files/laplace_agent_rooms/one_room/q_te.json", 0, 3, 500000, "Q - TE"),
    #             ("experiment/config_files/laplace_agent_rooms/one_room/q_to.json", 1, 3, 500000, "Q - TO"),
    #             ("experiment/config_files/laplace_agent_rooms/one_room/q_teo.json", 2, 3, 500000, "Q - TEO")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/laplace_agent_rooms/one_room/plot_onelayer.pdf")
    #
    # # Two Rooms
    # settings = [("experiment/config_files/laplace_agent_rooms_head/two_rooms/q.json", 1, 3, 800000, "Q"),
    #             ("experiment/config_files/laplace_agent_rooms_head/two_rooms/q_e.json", 0, 3, 800000, "Q - E"),
    #             ("experiment/config_files/laplace_agent_rooms_head/two_rooms/q_o.json", 3, 3, 800000, "Q - O"),
    #             ("experiment/config_files/laplace_agent_rooms_head/two_rooms/q_t.json", 1, 3, 800000, "Q - T"),
    #             ("experiment/config_files/laplace_agent_rooms_head/two_rooms/q_eo.json", 2, 3, 800000, "Q - EO"),
    #             ("experiment/config_files/laplace_agent_rooms_head/two_rooms/q_te.json", 1, 3, 800000, "Q - TE"),
    #             ("experiment/config_files/laplace_agent_rooms_head/two_rooms/q_to.json", 3, 3, 800000, "Q - TO"),
    #             ("experiment/config_files/laplace_agent_rooms_head/two_rooms/q_teo.json", 2, 3, 800000, "Q - TEO")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/laplace_agent_rooms_head/two_rooms/plot_twolayer.pdf")
    #
    # settings = [("experiment/config_files/laplace_linear/two_rooms/q.json", 0, 3, 800000, "Q"),
    #             ("experiment/config_files/laplace_linear/two_rooms/q_e.json", 0, 3, 800000, "Q - E"),
    #             ("experiment/config_files/laplace_linear/two_rooms/q_o.json", 2, 3, 800000, "Q - O"),
    #             ("experiment/config_files/laplace_linear/two_rooms/q_t.json", 0, 3, 800000, "Q - T"),
    #             ("experiment/config_files/laplace_linear/two_rooms/q_eo.json", 1, 3, 800000, "Q - EO"),
    #             ("experiment/config_files/laplace_linear/two_rooms/q_te.json", 0, 3, 800000, "Q - TE"),
    #             ("experiment/config_files/laplace_linear/two_rooms/q_to.json", 2, 3, 800000, "Q - TO"),
    #             ("experiment/config_files/laplace_linear/two_rooms/q_teo.json", 1, 3, 800000, "Q - TEO")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/laplace_linear/two_rooms/plot_linear.pdf")
    #
    #
    # # Hard Maze
    # settings = [("experiment/config_files/laplace_agent_rooms_head/hard_maze/q.json", 0, 3, 1000000, "Q"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze/q_e.json", 0, 3, 1000000, "Q - E"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze/q_o.json", 3, 3, 1000000, "Q - O"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze/q_t.json", 0, 3, 1000000, "Q - T"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze/q_eo.json", 3, 3, 1000000, "Q - EO"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze/q_te.json", 0, 3, 1000000, "Q - TE"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze/q_to.json", 3, 3, 1000000, "Q - TO"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze/q_teo.json", 2, 3, 1000000, "Q - TEO")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/laplace_agent_rooms_head/hard_maze/plot_twolayer.pdf")
    #
    # # Hard Maze
    # settings = [("experiment/config_files/laplace_linear/hard_maze/q.json", 0, 3, 1000000, "Q"),
    #             ("experiment/config_files/laplace_linear/hard_maze/q_e.json", 0, 3, 1000000, "Q - E"),
    #             ("experiment/config_files/laplace_linear/hard_maze/q_o.json", 1, 3, 1000000, "Q - O"),
    #             ("experiment/config_files/laplace_linear/hard_maze/q_t.json", 0, 3, 1000000, "Q - T"),
    #             ("experiment/config_files/laplace_linear/hard_maze/q_eo.json", 2, 3, 1000000, "Q - EO"),
    #             ("experiment/config_files/laplace_linear/hard_maze/q_te.json", 0, 3, 1000000, "Q - TE"),
    #             ("experiment/config_files/laplace_linear/hard_maze/q_to.json", 2, 3, 1000000, "Q - TO"),
    #             ("experiment/config_files/laplace_linear/hard_maze/q_teo.json", 1, 3, 1000000, "Q - TEO")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/laplace_linear/hard_maze/plot_linear.pdf")

    # settings = [("experiment/config_files/laplace_linear/hard_maze_2/q.json", 0, 3, 1000000, "Q"),
    #             ("experiment/config_files/laplace_linear/hard_maze_2/q_e.json", 0, 3, 1000000, "Q - E"),
    #             ("experiment/config_files/laplace_linear/hard_maze_2/q_o.json", 2, 3, 1000000, "Q - O"),
    #             ("experiment/config_files/laplace_linear/hard_maze_2/q_t.json", 0, 3, 1000000, "Q - T"),
    #             ("experiment/config_files/laplace_linear/hard_maze_2/q_eo.json", 3, 3, 1000000, "Q - EO"),
    #             ("experiment/config_files/laplace_linear/hard_maze_2/q_te.json", 0, 3, 1000000, "Q - TE"),
    #             ("experiment/config_files/laplace_linear/hard_maze_2/q_to.json", 1, 3, 1000000, "Q - TO"),
    #             ("experiment/config_files/laplace_linear/hard_maze_2/q_teo.json", 1, 3, 1000000, "Q - TEO")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/laplace_linear/hard_maze/plot_linear_2.pdf")


    # settings = [("experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q.json", 0, 3, 1000000, "Q"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_e.json", 0, 3, 1000000, "Q - E"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_o.json", 1, 3, 1000000, "Q - O"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_t.json", 3, 3, 1000000, "Q - T"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_eo.json", 3, 3, 1000000, "Q - EO"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_te.json", 0, 3, 1000000, "Q - TE"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_to.json", 2, 3, 1000000, "Q - TO"),
    #             ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_teo.json", 2, 3, 1000000, "Q - TEO")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/laplace_agent_rooms_head/hard_maze/plot_twolayer_2.pdf")

    settings = [("experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q.json", 0, 3, 1000000, "Q"),
                ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_e.json", 0, 3, 1000000, "Q - E"),
                ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_o.json", 0, 3, 1000000, "Q - O"),
                ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_t.json", 0, 3, 1000000, "Q - T"),
                ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_eo.json", 3, 3, 1000000, "Q - EO"),
                ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_te.json", 0, 3, 1000000, "Q - TE"),
                ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_to.json", 2, 3, 1000000, "Q - TO"),
                ("experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_teo.json", 2, 3, 1000000, "Q - TEO")]

    draw_lunar_lander_dqn(settings, save_path="plots/laplace_agent_rooms_head/hard_maze/plot_twolayer_2_1l.pdf")

