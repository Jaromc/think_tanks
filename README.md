# Genetic algorithm training tanks with neural networks to move #

Tanks enter an arena with the task of collecting as many targets as they can within a time period. The winners are used to spawn a new generation of tanks that can hopefully move better than the last. The whole process is hands off with the GA used to train and update the tanks. The tanks are initialised with random values initially so the first few generations don't move well but their intelligence picks up fairly quickly.

The model has a single input, rotation to target. It has 2 outputs which control how the tracks move on the tank.

The simulation is rendered to the console so training can be easily visualized.

### Hardware ###

Nvidia 3060 12GB

### Setup and Run ###

* docker build -t think_tanks .
* docker run --gpus all -it --rm -v /mnt/d/dev/think_tanks:/app think_tanks bash
* python3 train.py

### Demo ###
https://github.com/Jaromc/think_tanks/assets/89912906/d7f14ac3-7b88-455c-8444-b1d6ac14dc0d

### Learning Rate ###
![score_graph](https://github.com/Jaromc/think_tanks/assets/89912906/f2b0dedb-f1d9-436a-9489-5c42e453ef92)

### References ###
* https://subscription.packtpub.com/book/data/9781786462169/11/ch11lvl1sec84/working-with-a-genetic-algorithm
