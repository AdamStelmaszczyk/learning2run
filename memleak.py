from datetime import datetime
from os import getpid

from osim.env.run import RunEnv
from psutil import Process


def time():
    return datetime.now().strftime('%H:%M:%S')


def memory_used():
    process = Process(getpid())
    return process.memory_info().rss  # https://pythonhosted.org/psutil/#psutil.Process.memory_info


env = RunEnv(visualize=False)
env.reset()
step = 0
episode = 0
while True:
    observation, reward, done, info = env.step(env.action_space.sample())
    step += 1
    if done:
        episode += 1
        env.reset()
        print("%s Episode %s, steps %s, memory %s" % (time(), episode, step, memory_used()))
        step = 0
