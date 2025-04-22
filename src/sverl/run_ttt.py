from tictactoe_gym.envs.tictactoe_env import TicTacToeEnv

env = TicTacToeEnv()
env.reset()

env.step(0)
lst = env.step(7)


for i, ele in enumerate(lst):
    print(f"Ele {i}:", ele)
