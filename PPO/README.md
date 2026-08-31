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

Note:
r = ratio between new/old weights
gradient = entire networks gradients, mainly talking about what happens from one observation/action/reward sequence.
The (1 - e) and (1 + e) parts are only supposed to clip the change when the action is good/bad and you want to make it more/less likely.
So if an action was good and you want to make it more likely (r > 1), 1 + e would clip the ratio from exceeding 1.2.
And if an action was bad and you want to make it less likely, (0 > r > -1), 1 - e would clip it too.
The clipping completely cuts off the gradient (makes it 0), so there exists zero benefit towards increasing the ratio of this action anymore
When the gradient goes in a wrong way though, the gradient is still there, which allows the optimizer to change the gradient to go in the right direction still. Another explanation below if this doesn't make sense.

If you tried to change the weights incorrectly (increase bad decision or decrease good decision) (0 < r < 1 | r < -1), there would be no clipping at all, which would allow r to be as far from -1/1 as needed, but would also keep the gradient of the change, which would allow the optimizer to reverse the change.

Ultimately, the clip also cuts off the gradient, not just constraining the bounds, so the optimizer doesn't overassign specific policy changes.

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
