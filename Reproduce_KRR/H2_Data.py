#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jonpvandermause
"""

H2_ens = '/Users/jonpvandermause/Downloads/h2/energies.txt'
H2_dens = '/Users/jonpvandermause/Downloads/h2/densities.txt'

# read and store energies
# 500 in total
ens = []
with open(H2_ens, 'r') as outf:
    for line in outf:
        ens.append(float(line))
            
# read and store densities
dens = []
counter = 0
with open(H2_dens, 'r') as outf:
    for line in outf:
        dens.append([float(n) for n in line.split()])
        
        # print progress
        # done when 500 reached
        print(counter)
        counter+=1