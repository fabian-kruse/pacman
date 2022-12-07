# Deep Reinforcement Learning for Google Pacman

This javascript project implements a deep reinforcement learning agent capable of playing Google Pac-Man.
 
<div id="top"></div>


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The goal of this project is to develop and train an agent that to be capable of beating [Google Pac-Man](https://www.google.com/logos/2010/pacman10-i.html).

Because the state space of Pac-Man is rather large, this project works with features extracted from the game state.

This allows the agent to work with only an abstract version of the state space that is much smaller, which hopefully would lead to faster convergence towards good behaviour of the agent.

### Background 

#### Reinforcement Learning (RL)

Reinforcement learning describes a process in which an agent is taught to take actions in an environment that in order to maximize cumulative reward. It stands in contrast to supervised and unsupervised learning techniques, as the agent tries to learn good behaviour in the sense of maximizing cumulative reward by interacting with an environment and receiving corresponding rewards instead of learning from pre-existing data. Notice that for reinforcement learning approaches, the agent is completely unaware of its surrounding environment and is only given information about its current state and a set of different actions to take in that state. After the agent has taken an action, it is given a positive or negative reward from its environment. By generating some internal value system for each state, the agent is capable of learning which states are favorable or unfavorable i.e. give (comparatively) high or low reward and therefore are worthy or unworthy to visit again.

An important formula that captures the relation between the reward received and the internal value of a state is called [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation) and is given as follows:

$$U(s) = R(s) + \gamma \sum_{s'} P(s' \mid s,a) U(s')$$

A state $s$ has successors $s'$ that can be reached via action. The reward at state $s$ is given by $R(s)$ and $U(s)$ symbolizes the internal value of a state called utility value. It is simply given by the current reward plus a weighted sum over the successors of state $s$ with their respective transition probabilities, which is discounted by factor $\gamma \in \[0,1\]$. This **discount factor** $\gamma$ is a parameter that needs to be specified prior to the learning process and represents how much the agent should value reward that is potentially far in the future. 

#### Temporal-difference Learning (TD)

One way of using the Bellman equation in the learning process is called temporal-difference learning. While the agent interacts with its environment, the temporal-difference approach adjusts the utility values of observed states such that they agree with the Bellman equation. Therefore, the Bellman equation acts as a constraint equation that needs to be satisfied. 

By simply rewriting the equation we obtain the following update rule which needs to be applied once a transition occurs from state $s$ to $s'$ : 

$$U(s) \leftarrow U(s) + \alpha (R(s) + \gamma U(s') - U(s)) $$

The parameter $\alpha \in \(0,1\]$ is called **learning-rate** and determines to what extent newly acquired information overrides old information. 

Notice that utility values are only capable of capturing the value of a specific state, however they are not able to differentiate between different actions in a state. Depending on the action taken in a state the resulting successor state might have completely different utility values and using only utility values this can not be captured accurately.

Therefore, a natural extension of utility values $U(s)$ are quality values $Q(s,a)$ which consider actions seperately. 

They are related as follows:

$$U(s) = \max_{a} Q(s,a)$$

By substituting this relation into the previously mentioned update rule, we a reinforcement learning algorithm called **Q-learning** :

$$Q(s,a) \leftarrow Q(s,a) + \alpha ( R(s) + \gamma \max_{a'} Q(s',a') - Q(s,a))$$

This update rule is applied whenever action $a$ is executed in state $s$ leading to state $s'$.

Different approaches which do not make use of the difference between consecutive states first need to learn the underlying transition probabilities mentioned in the Bellman equation. However, temporal-difference learners are independent of these transition probabilities and are therefore called **model-free** as they do not need to learn the model ( $\rightarrow$ transition probabilities). This is one of the biggest advantage of the temporal-difference approach and the reason for their popularity.

#### Generalized Reinforcement Learning

In order to learn the Q-values for all states in the state space, the agent has to visit each states many times such that it is capable of reflecting their relation in the state space and while this is completely fine for small state spaces, it is infeasible for large spaces. 

One way to handle such problems is to use **function approximation** with **features**. 
Features are function handcrafted by the programmer and map states to (ideally) broader information. For the example of Pac-Man, a feature might represent the distance from that state to the closest ghost or to the next foodpill. A set of fueatures can then be used to approximate the <em>true</em> utility/quality function.

In the simplest case, the approximation function is a linear function with features $f_1, \dots, f_n$ and weights $\theta_1, \dots \theta_n$:
$$\hat{U}_\theta (s) = \theta_1 f_1(s) + \dots + \theta_n f_n(s)$$

In contrast to learning an individual value for each state, it suffices to learn values for $\theta_i$ such that the evaluation function $\hat{U}$ approximates the true utility function. This generalized reinforcement learning approach implies an enormous reduction of characterizing values that need to be learned are reduced from the number of states in the search space to a small number of weights. Further, this approach allows the agent to generalize from visited states to states it has not visited yet.

These ideas can easily be applied to temporal-difference learners by simply changing the update formula to update the weights instead of utility values: 

$$\theta_i  \leftarrow \theta_i + \alpha \[R(s) + \gamma \hat{U}_\theta (s') - \hat{U}_\theta (s) \] \frac{\partial \hat{U}_\theta (s)}{\partial \theta_i} $$

Notice, that we only used utility values here, but the notion of features can easily be extended to quality values, where features are functions of states <em>and </em> actions.

#### Deep Reinforcement Learning

Instead of approximating the true utility function with a fixed function (as described above using a linear function), we could extend the function approximation approach towards using neural networks instead. 
One important properties of neural network is the fact that under specific assumptions they can be used to approximate almost arbitrary functions, hence they are called **Universal function approximators**. 
This property makes them particularly useful for reinforcement learning. 

One approach is called **Actor Critic Method**, which involves using two neural networks, the "critic" and the "actor". Both networks are fed some information about the visited states from which the "critic"-network estimates the (quality/utility) value function, whereas the "actor"-network keeps track of an action distribution suggested by the "critic". In that way, the "critic" assists the "actor" in learning.

#### Exploration and Exploitation

A reinforcement learning agent tries to maximize the collected reward over (possibly infinite) time. Since it does not fully know the environment but only a subset of states, we need to take into consideration that the agent might not act optimally despite choosing actions that lead to high utility values or receiving high immediate rewards. This means that the agent can act optimally with respect to its current belief of the environment (the **model**), but suboptimal with respect to the true environment. Therefore, it is desirable to constantly improve the model to receive greater rewards in the future, while still collecting high rewards in the meantime. 

This problem is described by the **Exploration-Exploitation tradeoff** and can be tackled by an agent that explores new states forever while being greedy in the limit, i.e. chooses actions that bring optimal reward.

Common approaches are for example an $\epsilon$-greedy scheme in which the agent chooses a random action with a probability of $1-\epsilon$ and is greedy (chooses the action that leads to high values states) otherwise. Typically $\epsilon$ is value around $0.9$.

Other approaches include a probabilistic treatment to action selection where the action uses a softmax function.

For quality values, the action probabilities are given as follows:

$$\pi_\theta(s,a) =  \frac{e^{ \hat{Q_\theta} (s,a)}} {\sum_{a'} e^{ \hat{Q}_{\theta} (s,a')}} $$

<p align="right">(<a href="#top">back to top</a>)</p>

## Pac-Man

Pac-Man is played on a 2D-grid with a specific layout of walls, which makes the playfield a specific labyrinth. 
During the game, Pac-Man tries to eat all foodpills spread out across the playfield before being hunted down by one of the four ghost that are following him.

Further, there is a small number of powerpills that help Pac-Man beat the ghosts. After Pac-Man eats such a pill, the ghosts are frightened for a short time and try to run away from Pac-Man. Only during that time Pac-Man is able to eat the ghost which brings them back to their home.
Once Pac-Man eats all pills before being hunted down three times, he beats that level.

In recognition to its thirty year anniversary in may 2010 Google changed its logo to a playable version of Pac-Man. This version is still available as doodle and can be found [here](https://www.google.com/doodles/30th-anniversary-of-pac-man). 

<p align="right">(<a href="#top">back to top</a>)</p>

## PROJECT DETAILS

This project implements two different kinds of agent to interact with the environment of Google Pac-Man. The goal of the agents is to complete the first level.

The first, simpler agent is based on linear function approximation, whereas the second, more sophisticated agent uses a neural network to approximate the true utility function. 

Both agents use the same set of features described below. 


### Features
The following set of features are used in this project to give the agent a general valuation of a specific state. 

All features are normalized to have values in $\[0,1\]$ where $0$ is the worst and $1$ the best possible value (for the agent) of a feature.

1. Progress of level 
2. Progress of eaten powerpill
3. Distance to next foodpill
4. Distance to next unfrightened ghost
5. Distance to next frightened ghost
6. Distance to next powerpill
7. Evaluate if current action is opposite of previous action

### Reward
A central element in reinforcement learning is which states earn what amount of reward, and to which events to these states translate.

For this project, the following set of rewards is used:

1. Agent completes level $\rightarrow 500$
2. Agent is eaten by ghost $\rightarrow -300$
3. Agent ate an unfrightened ghost $\rightarrow 20$
4. Agent ate foodpill $\rightarrow 12$
5. Agent ate a powerpill $\rightarrow 5$
6. Agent reverse moving direction $\rightarrow -8$

### Agents

#### Linear Function Agent
The linear function appoximator uses a Q-learning approach to approximate the true Q-function using a linear function. 

Given the set of features for each state, it updates its internal weights using update rules as described above. 

For that is uses a learning rate of $\alpha = 0.1$, a discount factor of $\gamma = 0.5$. 
Further, it employs an $\epsilon$-greedy action selection scheme where $\epsilon = 0.9$, so it chooses a random action in about every tenth state and is greedy otherwise.

#### Actor-Critic Agent

The actor-critic agent uses two different neural networks to estimate the state value and the action distribution separately.

The "actor"-network consists of the input layer that takes the features as input. The hidden layer consists out of 50 hidden nodes which are then linked to the output layer with one node for each action. Because the output values are logits, it uses a soft max cross entropy to calculate the loss between the prediction and the target.

The "critic"-network consists out of seven input nodes, 50 hidden nodes, but only a single output node representing the estimated quality value of the input state.
Since the output is a real number and not a logit, it uses a mean squared error loss.

In contrast to the linear function agent, the actor-critic agent uses a softmax action selection scheme. This makes it such that action that lead to states with low state values are taken proportionally lower.

### Built With

* [TensorFlow.js](https://www.tensorflow.org/js)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Fabian Kruse - fabian_kruse@gmx.de

Project Link: [https://github.com/fabian-kruse/pacman](https://github.com/fabian-kruse/pacman)

<p align="right">(<a href="#top">back to top</a>)</p>


© 2022 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
