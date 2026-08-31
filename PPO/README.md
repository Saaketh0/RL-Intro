PPO has the same overall objective as TRPO, to lessen the policy change to prevent big changes. Statistically, a bunch of smaller changes would converge faster/better than big steps, as a small mistake could be catastropic in the latter but just inconvenience in the former.

TRPO addresses this by calculating the KL Divergence and Hessian, which requires a bunch of computation, but we can get a similar result with much less computation with PPO, simply clipping the policy change.

So how is PPO different from TRPO? Despite being done after TRPO, PPO is much much simpler. In fact, below I will be putting all the lines of code I added to change the algo from VPA to PPO below:

```
self.epsilon = 0.2
unclipped = advantage * new_over_old
clipped = advantage * torch.clamp(new_over_old, 1-self.epsilon, 1+self.epsilon)
l = torch.min(unclipped, clipped)

total_error = -l.mean() # This line was already in VPA, just l was unclipped

# Below are the lines already in VPA, the backwards and step steps.
total_error.backward()
self.optimizer.step()
```

# Clipped this from ChatGPT, think it explains it again nicely if the explanation above wasn't good.

As you can see below, all we are going is cutting off any gradient that exceeds a 20% change either up or down.

If PPO uses:

$$ \epsilon=0.2 $$

then the clipping boundaries are:

$$ 1-\epsilon=0.8,\qquad 1+\epsilon=1.2 $$

So:

$$ 0.8 \le r \le 1.2 $$

means:

PPO considers up to roughly a 20% change in the probability of that action before the clipping mechanism starts mattering.
