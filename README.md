# Adaptive Importance Sampling for Bayesian Networks

Note that this currently only works for nodes with two states, although
some work has been done to generalize it to networks with more states.
This is a (loose) implementation of AIS-BN by Cheng, Druzdzel.
See: https://arxiv.org/abs/1106.0253

I couldn't find code I was happy with, so I wrote this instead. 
Hope it's useful.


*Note*: This example considers a noisy-OR parametrization and will run, but I accidentally broke the updating of the proposals. I may fix it, if I have some time. 
