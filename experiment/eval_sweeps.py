import os
import numpy as np
import sys
import argparse

sys.path.append(os.path.abspath('../'))

from sweeper import Sweeper


def extract_line(lines, max_steps, interval=0):
    """
    Retrieves num_steps at which episodes were completed
    and returns an array where the value at index i represents
    the number of episodes completed until step i
    """
    step = 0
    # rewards_over_time = np.zeros(max_steps//interval+1) # for GANModel training
    rewards_over_time = np.zeros(max_steps // interval) # for DQN training
    try:
        for line in lines:
            if 'total steps' not in line:
                continue
            num_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            if num_steps > max_steps:
                break
            # reward = float(line.split("|")[1].split(",")[1].split(" ")[2]) # for GANModel training
            reward = float(line.split("|")[1].split(",")[2].split("/")[0].split(" ")[-1]) # for DQN training
            rewards_over_time[step] = reward
            step += 1
        return rewards_over_time
    except:
        print(line)
        print('step:{}'.format(step))
        raise


def get_max_steps(lines):
    for line in lines[::-1]:
        if 'total steps' in line:
            max_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            return max_steps
    return -1


def _eval_lines(config_file, start_idx, end_idx, max_steps, interval=10000):
    print('config_file: {}'.format(config_file))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sweeper = Sweeper(os.path.join(project_root, config_file))
    eval = []
    eval_lines = []
    for k in range(sweeper.total_combinations):
        eval.append([])
        eval_lines.append([])

    for idx in range(start_idx, end_idx):
        cfg = sweeper.parse(idx)
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        log_dir = cfg.get_logdir()
        log_path = os.path.join(log_dir, 'log')
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            continue
        if len(lines) == 0:
            continue
        # ugly parse based on the log_file format
        try:
            num_steps = get_max_steps(lines)
            if num_steps >= max_steps:
                assert idx % sweeper.total_combinations == cfg.param_setting
                avg_eval_steps = extract_line(lines, max_steps, interval=interval)
                eval[idx % sweeper.total_combinations].append(np.mean(avg_eval_steps[-int(len(avg_eval_steps)/2):]))

        except IndexError:
            print(idx)
            raise
    summary = list(map(lambda x: (x[0], np.mean(x[1]), np.std(x[1]), len(x[1])), enumerate(eval)))
    summary = [x for x in summary if np.isnan(x[1]) == False]
    # new_summary = []
    # for s in summary:
    #     if np.isnan(s[1]) == False:
    #         new_summary.append(s)
    # print(summary[0])
    # print(new_summary[0])
    # quit()

    summary = sorted(summary, key=lambda s: s[1], reverse=True)

    for idx, mean, std, num_runs in summary:
        print("Param Setting # {:>3d} | Rewards: {:>10.10f} +/- {:>5.2f} ({:>2d} runs) {} | ".format(
            idx, mean, std, num_runs, sweeper.param_setting_from_id(idx)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--config-file', default='sweep.json')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=72, type=int)
    parser.add_argument('--max-steps', default=50000, type=int)
    parser.add_argument('--interval', default=10, type=int)

    args = parser.parse_args()

    # ONE ROOM
    # print("q")
    # _eval_lines(config_file='experiment/config_files/rooms/one_room/q.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/rooms/one_room/q_e.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/rooms/one_room/q_eo.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/rooms/one_room/q_o.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/rooms/one_room/q_t.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/rooms/one_room/q_te.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/rooms/one_room/q_teo.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/rooms/one_room/q_to.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te2o")
    # _eval_lines(config_file='experiment/config_files/rooms/one_room/q_te2o.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te3o")
    # _eval_lines(config_file='experiment/config_files/rooms/one_room/q_te3o.json', start_idx=0, end_idx=12, max_steps=500000)

    # TWO ROOMS
    # print("q")
    # _eval_lines(config_file='experiment/config_files/rooms/two_rooms/q.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/rooms/two_rooms/q_e.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/rooms/two_rooms/q_eo.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/rooms/two_rooms/q_o.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/rooms/two_rooms/q_t.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/rooms/two_rooms/q_te.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/rooms/two_rooms/q_teo.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/rooms/two_rooms/q_to.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te2o")
    # _eval_lines(config_file='experiment/config_files/rooms/two_rooms/q_te2o.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te3o")
    # _eval_lines(config_file='experiment/config_files/rooms/two_rooms/q_te3o.json', start_idx=0, end_idx=12, max_steps=500000)

    # Hard Maze
    # print("q")
    # _eval_lines(config_file='experiment/config_files/rooms/hard_maze/q.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/rooms/hard_maze/q_e.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/rooms/hard_maze/q_eo.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/rooms/hard_maze/q_o.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/rooms/hard_maze/q_t.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/rooms/hard_maze/q_te.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/rooms/hard_maze/q_teo.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/rooms/hard_maze/q_to.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te2o")
    # _eval_lines(config_file='experiment/config_files/rooms/hard_maze/q_te2o.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te3o")
    # _eval_lines(config_file='experiment/config_files/rooms/hard_maze/q_te3o.json', start_idx=0, end_idx=12, max_steps=500000)

    # ONE ROOM
    # print("q")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/one_room/q.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/one_room/q_e.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/one_room/q_o.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/one_room/q_t.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/one_room/q_eo.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/one_room/q_te.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/one_room/q_to.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/one_room/q_teo.json', start_idx=0, end_idx=12, max_steps=500000)


    # TWO ROOMS
    # print("q")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/two_rooms/q.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/two_rooms/q_e.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/two_rooms/q_o.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/two_rooms/q_t.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/two_rooms/q_eo.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/two_rooms/q_te.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/two_rooms/q_to.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/two_rooms/q_teo.json', start_idx=0, end_idx=12, max_steps=500000)

    # Hard Maze
    # print("q")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/hard_maze/q.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/hard_maze/q_e.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/hard_maze/q_o.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/hard_maze/q_t.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/hard_maze/q_eo.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/hard_maze/q_te.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/hard_maze/q_to.json', start_idx=0, end_idx=12, max_steps=500000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms/hard_maze/q_teo.json', start_idx=0, end_idx=12, max_steps=500000)

    # TWO ROOMS
    # print("q")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/two_rooms/q.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/two_rooms/q_e.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/two_rooms/q_o.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/two_rooms/q_t.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/two_rooms/q_eo.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/two_rooms/q_te.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/two_rooms/q_to.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/two_rooms/q_teo.json', start_idx=0, end_idx=12, max_steps=800000)

    # print("q")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/two_rooms/q.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/two_rooms/q_e.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/two_rooms/q_o.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/two_rooms/q_t.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/two_rooms/q_eo.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/two_rooms/q_te.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/two_rooms/q_to.json', start_idx=0, end_idx=12, max_steps=800000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/two_rooms/q_teo.json', start_idx=0, end_idx=12, max_steps=800000)

    # # Hard Maze
    # print("q")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze/q.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze/q_e.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze/q_o.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze/q_t.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze/q_eo.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze/q_te.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze/q_to.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze/q_teo.json', start_idx=0, end_idx=12, max_steps=1000000)

    # Hard Maze
    # print("q")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze/q.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze/q_e.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze/q_o.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze/q_t.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze/q_eo.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze/q_te.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze/q_to.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze/q_teo.json', start_idx=0, end_idx=12, max_steps=1000000)

    # print("q")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze_2/q.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze_2/q_e.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze_2/q_o.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze_2/q_t.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze_2/q_eo.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze_2/q_te.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze_2/q_to.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/laplace_linear/hard_maze_2/q_teo.json', start_idx=0, end_idx=12, max_steps=1000000)

    # # Hard Maze
    # print("q")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_e")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_e.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_o")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_o.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_t")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_t.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_eo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_eo.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_te")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_te.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_to")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_to.json', start_idx=0, end_idx=12, max_steps=1000000)
    # print("q_teo")
    # _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2/q_teo.json', start_idx=0, end_idx=12, max_steps=1000000)

    print("q")
    _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q.json', start_idx=0, end_idx=12, max_steps=1000000)
    print("q_e")
    _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_e.json', start_idx=0, end_idx=12, max_steps=1000000)
    print("q_o")
    _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_o.json', start_idx=0, end_idx=12, max_steps=1000000)
    print("q_t")
    _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_t.json', start_idx=0, end_idx=12, max_steps=1000000)
    print("q_eo")
    _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_eo.json', start_idx=0, end_idx=12, max_steps=1000000)
    print("q_te")
    _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_te.json', start_idx=0, end_idx=12, max_steps=1000000)
    print("q_to")
    _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_to.json', start_idx=0, end_idx=12, max_steps=1000000)
    print("q_teo")
    _eval_lines(config_file='experiment/config_files/laplace_agent_rooms_head/hard_maze_2_1l/q_teo.json', start_idx=0, end_idx=12, max_steps=1000000)

