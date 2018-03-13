# Find-the-bot
It's a Kaggle competition for Machine Learning Data Science (CS5780).

Background: A bot moves around in a 2D plane following some probabilistic pattern unknown to you. You don't observe this bot's location on every time-step. What you do observe is the angle of the box to x axis at every time step. On every run, we place the bot at its starting location (fixed at same starting location for all runs) and let it run for 1000 +1 steps. We perform 10000 runs each with 1000+1 steps.
You are given observations for 10000 runs of the angle observed at each step. You are additionally provided exact location of the bot at some random time steps on every round for the first 6000 rounds only.

Goal:  predict the final location of the bot at the 1001'th step for rounds 6001 to 10000.

Implement: run kernelizedSVMs.py

Kaggle Link: https://www.kaggle.com/c/competition-2-cs4786-fall-17
