# Attempting to build an RL agent to play snake.

Currently can build a dense feed forward network to beat non-growing snake on a 3x3 field.

Non-growing snake is essentially gridworld and *does not* need a estimator to capture all q values, but it is
for learning.

Training was working slower for a 10x10 and the snake still gets in loops.

Next steps are CNN on this space _or_ sparse reward mitigation techniques.

Built against TF 2.3 with tf-agents[reverb]
