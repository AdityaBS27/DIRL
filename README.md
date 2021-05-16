# DIRL
Vehicle Trajectory prediction using inverse reinforcement learning

The goal of the project is to achieve a suitable prediction model of driver’s intention,
in case of lane change scenarios to help in improving ADAS systems. The prediction
models must be robust to predict varied situations. Consider a highway driving
scenario the ego vehicle needs to predict the surrounding vehicle intentions, whether
the vehicle would allow the ego vehicle to pass, does the vehicle reduce the speed and
many such characteristics features of the surrounding vehicle. Our aim is analyze various
behavior of the surrounding vehicle during lane change scenarios for prediction.
The ego vehicle, the surrounding vehicle and their interacting behaviour during
lane change are considered experts agents. These expert agents behave optimally
during the lane change. The expert’s behaviour are considered to be MDP, whose output
is a time series state trajectories, that are extracted form HighD.
These extracted trajectories are considered to be expert trajectories . The
task is to analyze all these trajectories ¿E of the surrounding vehicle in the data and
build a model to predict such driving patterns for future reference to the ADAS. So IRL
techniques are employed to understand the reward structure of this expert trajectories. Based on on the reward the driving behaviour of the surrounding vehicle are
predicted.
