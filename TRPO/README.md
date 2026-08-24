So this is the next algorithm being implemented after REINFORCE.

The Trust Region part of TRPO is the addition of a new bounding metric called KL divergence, which basically caps the change in the policy's action distribution. So under the same environment, if the new action outputs are too different, KL doesn't like that.
The weights in the NN can change alot, but as long as the outputs they give don't change too much, its fine.
- That way, no big thing is able to drastically change the action distribution all at once.
- DKL​(πold​∥πnew​)≤δ

The "new" policy network is just the NN, but not frozen, so it contains variables that are meant to change, with the graph PyTorch stores of the NN attached to for eventual gradient changes.

Plain KL would tell you: How different are these two policies
Hessian KL would tell you: If the weights are changed in this way, how quickly would KL increase
Hessian KL is taking the second order derivative of the KL function, to be able to estimate the curvature

So what is the point of the Hessian? If we just take the gradients, g, and then bound it by the KL value, we will
get a decently good estimate, but would just be bounded by how different the policies are, rather than the best change that could happen in the step size.

Ok, so I will put the steps in TRPO down below.

1. Initialize a randomly initialized NN, running through a couple episodes and gathering data, no changes to NN yet.
2. When 10 episodes are done, you now would go to update weights, which would first calculate the discounted return, baseline, and advantage (explained in REINFORCE folder)
3. Then, you want to calculate the gradient of the changes that the updated (new) NN would get by:
  4. Copying and freezing the current NN weights, and taking the action logits of each observation
  5. Doing the same with the live weights (this and the old should be the same values, except the old ones are constant frozen values and these live ones are more akin to variables)
  6. Divinding the live action probabilities by the old frozen ones, and then multiplying by the advantage at that step.
  7. Taking the mean of all those values throughout the entire episode, and applying .backwards() on it.
8. Now you have a gradient of changes, "g", which would show a potential way of improving the weights.
  9. But what if this gradient is not the best direction, and what if this gradient is too big?
  10. Also, what if there is some way to get similar/better output results as this gradient changes, but also keeping within a limit of how much change can happen?
  11. This is what the KL Divergence step aims to solve
12. Compute the best gradient: x ~ H-1g, then scale it to satisfy the KL constraint (most code in this part has comments explaining it)
13. Update weights


Results: This converged much much faster than REINFORCE, hitting the 5k step limit within a minute of training unlike REINFORCE, which only when using JAX was able to get that kind of speed. (I know this is not the best measure, this was just my observation)


Big Confusions For Me.
So I got confused by a lot of small things that took a while for me to understand, so while some of these things may be stated above, I just wanted to leave a basic list here in case other people also get confused too.

- The new and old NN's are the same, the difference is the "new" is the actual NN, while the "old" are basically just numbers that the old NN outputted. ChatGPT quote: "But there is one crucial detail. The numerator is connected to the trainable weights, while the denominator is frozen."
- The magnitude of a weight change does not directly tell you the magnitude of the policy distribution change, so the KL divergence doesn't measure how much the weights shift at all, it is just measuring how these shifts alter the output.
- The equation we are given, "new/old" is exactly the same as REINFORCE's equation when old==new, and considering that they are the same during the entire update_weights, this is basically just REINFORCE + KL Divergence.
- x, is the update direction produced by transforming the gradient using the KL Hessian, it isn't g * some scalar, but rather a completely new gradient, that would achieve a better/same result as g, while keeping the change not too big.
  - ChatGPT quote: "g is the ordinary policy gradient. x = H^-1 g is a KL-aware update direction derived from that gradient."
