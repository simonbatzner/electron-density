#!/bin/bash
for i in 10 20 30 40 50 60 70 80 90 100
do
  python3 NN_hyperparam.py --hidden $i --lr 0.001 --epochs 50000
done
