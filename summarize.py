import argparse
import ast
import logging

import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend('agg')
matplotlib.style.use('ggplot')

import shutil
import os


def dirname(model):
    return model + '_plots'


def override_dir(model):
    d = dirname(model)
    print("Saving %s" % d)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def plot_diagrams(history, model):
    override_dir(model)
    for key, results in history.items():
        plot_diagram(history, key, "%s/%s.png" % (dirname(model), key))


def plot_diagram(history, key, filename):
    plt.clf()
    plt.plot(history[key])
    plt.xlabel('iteration')
    plt.ylabel(key)
    logging.info("Saving diagram: %s" % filename)
    plt.savefig(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create summary from training/test history')
    parser.add_argument('--model', dest='model', action='store', default="default")
    args = parser.parse_args()
    with open(args.model + '_history', 'r') as f:
        s = f.read()
        s = s.replace('nan', 'None')
        history = ast.literal_eval(s)
        plot_diagrams(history, args.model)
