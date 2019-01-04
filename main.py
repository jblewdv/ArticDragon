# *************************************************************************
#
# Parallax Capital INVESTMENTS
# __________________
#
#  Copyright (c) 2018 Joshua Blew
#  All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of Parallax Capital Investments and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Parallax Capital Investments
# and its suppliers and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Parallax Capital Investments.
# /

from __future__ import print_function
import os
import neat
import pandas as pd
import numpy as np
# import visualize
import matplotlib.pyplot as plt
import pickle

raw = pd.read_csv('../CVX.csv', usecols=(1,2,3,4))
data = [tuple(x) for x in raw.values]
data = data[::-1]
closes = []
for x in data:
    closes.append(x[3])

data = data[-250:]
closes = closes[-250:]

dayChange = 3
window = 20
expected_output = []

for id, i in enumerate(closes):
    if id < len(data) - dayChange:
        condition = False
        delta = (closes[id+dayChange] / i) - 1

        if delta > 0.02:
            expected_output.append(1)
            # plt.plot(id, i, 'go', markersize=5)
        else:
            expected_output.append(0)

def eval_genomes(genomes, config):

    for genome_id, genome in genomes:

        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        for i in range(window, len(data)-dayChange):
            inputs = []
            prices = data[i-window:i]

            for tuple in prices:
                for value in tuple:
                    inputs.append(value)

            output = net.activate(inputs)
            output = round(output[0], 0)

            if output == 1 & expected_output[i] == 1:
                genome.fitness += 1

def run(config_file):

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 20)

    with open('saved', 'wb') as f:
        pickle.dump(winner, f)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    trades = []

    for i in range(window, len(data)-dayChange):
        inputs = []
        prices = data[i-window:i]

        for tuple in prices:
            for value in tuple:
                inputs.append(value)

        output = winner_net.activate(inputs)
        output = round(output[0], 0)

        if output == 1 & expected_output[i] == 1:
            plt.plot(i, closes[i], 'go', markersize=5)

    plt.plot(closes, c='blue')
    plt.show()

    # node_names = {0:'No Trade', 1: 'BUY'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_file')
    run(config_path)
