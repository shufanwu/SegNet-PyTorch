#!/bin/bash
set -e

python weight_transfer.py
setsid python main.py --mode train>>train_message.txt &
#setsid python main.py --mode val>>train_message.txt &
