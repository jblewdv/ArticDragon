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
import pickle
import matplotlib.pyplot as plt

raw = pd.read_csv('../CVX.csv', usecols=(1,2,3,4))
data = [tuple(x) for x in raw.values]
data = data[::-1]
closes = []
for x in data:
    closes.append(x[3])

data = data[-500:-250]
closes = closes[-500:-250]

dayChange = 3
window = 20

acctValue = 100000
shares = 500
gains, losses, tradeCount = 0, 0, 0
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

with open('saved', 'rb') as f:
    c = pickle.load(f)

print ("Loaded Genome Successfully...")

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_file')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)
tradeID = 0

for i in range(window, len(data)-dayChange):
    inputs = []
    prices = data[i-window:i]

    for tuple in prices:
        for value in tuple:
            inputs.append(value)

    output = net.activate(inputs)
    output = round(output[0], 0)

    if i >= tradeID + dayChange:
        if output == 1 & expected_output[i] == 1:
            tradeCount += 1
            PnL = ((shares * data[i+dayChange][3]) - (shares * prices[-1][3]))
            acctValue += PnL
            if PnL > 0:
                gains += PnL
            else:
                losses += PnL

            plt.plot(i, closes[i], 'go', markersize=5)

print (acctValue, tradeCount, gains, losses)
plt.plot(closes, c='blue')
plt.show()
