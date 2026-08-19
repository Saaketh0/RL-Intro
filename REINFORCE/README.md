# The REINFORCE algorithms.

All of them have their own neural network, with the og and batched having the same exact one, and the JAX one only being changed to fit into Flax standards. 

They all also have a very basic baseline calculation, which is just the average of all discounted_returns recieved at that timestep.

REINFORCE has no speedups, and was purely made to be an implementation of REINFORCE
REINFORCE_BATCHED makes it so that you run all the weight change calculations in update_weights at once, see episodal batching below
Both functions above used the cartpole.py file to run it

REINFORCE_JAX is a whole separate beast, having converted all of its operations to run on GPU using the JAX and FLAX libraries. Uses cartpole_jax.py to work. By far the fastest at training




Two types of batching take place:
- Environment Batching: Using SyncEnv and Async env from the Gym API, you spawn multiple environments and run through them
- Episodal Batching: Since the NN doesn't change until the very end of the episode, you collect all the observations needed for calculations and at the very end, process all of those observations at once
