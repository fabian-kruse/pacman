# Deep Reinforcement Learning for Google Pacman

This javascript project implements a deep reinforcement learning agent capable of playing google pacman.
 
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

The goal of this project is to develop and train an agent that to be capable of beating [google pacman](https://www.google.com/logos/2010/pacman10-i.html).

Because the state space of pacman is rather large, this project works with features extracted from the game state. (Include list of features)

This allows the agent to work with only an abstract version of the state space that is much smaller, which hopefully would lead to faster convergence towards good behaviour of the agent.


<p align="right">(<a href="#top">back to top</a>)</p>

### Background 

#### Reinforcement Learning (RL)

Reinforcement learning describes a process in which an agent is taught to take actions in an environment that in order to maximize cummulative reward. It stands in contrast to supervised and unsupervised learning techniques, as the agent tries to learn good behaviour in the sense of maximizing cummulative reward by interacting with an environment and receiving corresponding rewards instead of learning from pre-existing data. Notice that for reinforcement learning approaches the agent is completely unaware of its surrounding environment and is only given information about its current state and a set of different actions to take in that state. After the agent has taken an action it is given a positive or negative reward from its environment. By generating some internal value system for each state, the agent is capable of learning which states are favorable or unfavorbale i.e. give (comparatively) high or low reward.

An important formula that captures the relation between the reward received and the internal value of a state is called [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation) and is given as follows:

$$U(s) = R(s) + \gamma \sum_{s'} P(s' \mid s,a) U(s')$$

A state $s$ has successors $s'$ that can be reached via action. The reward at state $s$ is given by $R(s)$ and $U(s)$ symbolizes the internal value of a state called utility value. It is simply given by the current reward plus a weighted sum over the successors of state $s$ with their respective transition probabilities which is discounted by factor $\gamma \in \[0,1\]$. This **discount factor** $\gamma$ is a parameter that needs to be specified prior to the learning process and represents how much the agent should value reward that potentially far in the future. 

#### Temporal-difference Learning (TD)

One way of using the Bellman equation (maybe reference) in the learning process is called temporal-difference learning. While the agent interacts with its environment, the temporal-difference approach adjusts the utility values of observed states such that they agree with the Bellman equation. Therefore, the Bellman equation acts as a constraint equation that needs to be satisfied. 

By simply rewriting the equation we obtain the following update rule which needs to be applied once a transtition occurs from state $s$ to $s'$ : 

$$U(s) \leftarrow U(s) + \alpha (R(s) + \gamma U(s') - U(s)) $$

The paramater $\alpha \in \(0,1\]$ is called **learning-rate** and determines to what extent newly acquired information overrides old information. 

Notice that in utility values are only capable of capturing the value of a specific state, however they are not able to differentiate between different actions in a state. Depending on the action taken in a state the resulting successor state might have completely different utility values and using only utility values this can not be captured accurately.

Therefore, a natural extension of utility values $U^\pi(s)$ are quality values $Q^\pi(s,a)$ which consider actions seperately. 

They are related as follows:

$$U(s) = \max_{a} Q(s,a)$$

By substituting this relation into the previously mentioned update rule, we a reinforcement learning algorithm called **Q-learning** :

$$Q(s,a) \leftarrow Q(s,a) + \alpha ( R(s) + \gamma \max_{a'} Q(s',a') - Q(s,a))$$

This update rule is applied whenever action $a$ is executed in state $s$ leading to state $s'$.

Different approaches which do not make use of the difference between consecutive states first need to learn the underlying transition probabilites mentionend in the Bellman equation. However, temporal-difference learners are independent of these transition probabilities and are therefore called **model-free** as they do not need to learn the model (->transition probabilities). This is one of the biggest advantage of the temporal-difference apporach and the reason for their popularity.

#### Generalized Reinforcement Learning

In order to learn the Q-values for all states in the state space, the agent has to visit each states many times such that it is capable of reflecting their relation in the state space and while this is completely fine for small state spaces, it is infeasible for large spaces. 

One way to handle such problems is to use **function approximation** with **features**. 
Features are function handcrafted by the programmer and map states to (ideally) broader information. For the example of pacman a feature might represent the distance fromt that state to the closest ghost or to the next foodpill. A set of fueatures can then be used to approximate the <em>true</em> utility/quality function.

In the simplest case, the approximation function is a linear function with features $f_1, \dots, f_n$ and weights $\theta_1, \dots \theta_n$:
$$\hat{U}_\theta (s) = \theta_1 f_1(s) + \dots + \theta_n f_n(s)$$

In contrast to learning an individual value for each state, it suffices to learn values for $\theta_i$ such that the evaluation function $\hat{U}$ approximates the true utility function. This generalized reinforcement learning approach implies an enourmous reduction of characterizing values that need to be learned are reduced from the number of states in the search space to a small number of weights. Further, this approach allows the agent to generalize from visited states to states it has not visited yet.

These ideas can easily be applied to temporal-differnce learners by simply changing the update formula to update the weights instead of utility values: 

$$\theta_i  \leftarrow \theta_i + \alpha \[R(s) + \gamma \hat{U}_\theta (s') - \hat{U}_\theta (s) \] \frac{\partial \hat{U}_\theta (s)}{\partial \theta_i} $$

Notice, that we only used utility values here, but the notion of features can easily be extended to quality values where features are functions of states <em>and </em> actions.

#### Exploration and Exploitation

A reinforcement learning agent tries to maximize the collected reward over (possibly infinite) time. Since it does not fully know the environment but only a subset of states, we need to take into consideration that the agent might not act optimally despite choosing actions that lead to high utility values or receiving high immediate rewards. This means that the agent can act optimally with respect to its current belief of the environment (the **model**), but suboptimal with respect to the true environment. Therefore it is desireable to constantly improve the model to receive greater rewards in the future, while still collecting high rewards in the meantime. 

This problem is described by the **Exploration-Exploitation tradeoff** and can be tackled by an agent that explores new states forever while being greedy in the limit i.e. chooses actions that bring optimal reward.

Common approaches are for example an $\epsilon$-greedy scheme in which the agent chooses a random action with a probability of $1-\epsilon$ and is greedy (chooses the action that leads to high values states) otherwise. Typically $\epsilon$ is value around $0.9$.

Other approaches include a probabilistic treatment to action selection where the action uses a softmax function.

For quality values the action probabilities are given as follows:

$$\pi_\theta(s,a) =  \frac{e^{ \hat{Q_\theta} (s,a)}} {\sum_{a'} e^{ \hat{Q}_{\theta} (s,a')}} $$

## PACMAN

Description of the game.

In recognition to its thirty year anniversary in may 2010 Google changed its logo to a playable version of pacman. This version is still available as doodle and can be found [here](https://www.google.com/doodles/30th-anniversary-of-pac-man).

## PROJECT DETAILS

This project implements two different kinds of agent to interact with the environment of google pacman. The goal of the agents is to complete the first level.

The first, simpler agents is based on linear function approximation whereas the second, more sophistacated agent uses a neural network to approximate the true utility function. 

Both agents use the same set of features described below. 


### Features
The following set of features are used in this project to give the agent a general valuation of a specific state. 

All feature are normalized to have values in $\[0,1\]$ where $0$ is the worst and $1$ the best possible value (for the agent) of a feature.

1. Progress of level 
2. Progress of eaten powerpill
3. Distance to next foodpill
4. Distance to next unfrightened ghost
5. Distance to next frightened ghost
6. Distance to next powerpill
7. Evaluate if current action is opposite of previous action

### Reward
A central element in reinforcement learning is which states earn what amount of reward and to which events to these states translate.

For this project the following set of rewards is used:

1. Agent completes level $\rightarrow 500$
2. Agent is eaten by ghost $\rightarrow -300$
3. Agent ate an unfirghtened ghost $\rightarrow 20$
4. Agent ate foodpill $\rightarrow 12$
5. Agent ate a powerpill $\rightarrow 5$
6. Agent reverse moving direction $\rightarrow -8$

### Agents

#### Linear function approximator

#### neural approximator



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
