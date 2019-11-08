# Reinforcement Learning For Optimal Trade Execution paper

This paper was written as part of the Applied Project of the MSc Risk Management & Financial Engineering at Imperial College London.

The paper aims to explore Value based, Deep Reinforcment Learning (Deep Q-Learning and Double Deep Q-Learning) for the problem of Optimal Trade Execution. The problem of Optimal Trade Execution aims to find the the optimal "path" of executing a stock order, or in other words the number of shares to be executed at different steps given a time constraint, such that the price impact from the market is minimised and consequently revenue from executing a stock order maximised.

The results of the paper shows that under a simple environment, where the optimal execution strategy is known (TWAP), the RL agent is able to very closely approximate this solution. As a consequence, this may serve as a proof of concept that under a broader environment the RL has the potential to learn more patterns relevant to the problem, like stock price prediction or external predatory trading, which would allow it to outperform the TWAP in practice. 

For more information please see the "AP Report - RL for Optimal Exectuion.pdf" regarding the paper and "deep_q_learning.py" regarding the code.
