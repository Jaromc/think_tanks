# Genetic algorithm training tanks with neural networks to move #

Tanks enter an arena with the task of collecting as many targets as they can within a time period. The winners are used to spawn a new generation of tanks that can hopefully move better than the last.

The model has a single input, rotation to target. It has 2 outputs which control how the tracks move on the tank.

The simulation is rendered to the console so training can be easily visualized.

### Setup and Run ###

* docker build -t think_tanks .
* docker run --gpus all -it --rm -v /mnt/d/dev/think_tanks:/app think_tanks bash
* python3 train.py

### References ###
* https://subscription.packtpub.com/book/data/9781786462169/11/ch11lvl1sec84/working-with-a-genetic-algorithm