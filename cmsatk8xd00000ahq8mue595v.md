---
title: "Implementing Algorithms for the Multi-Armed-Bandit Problem"
seoTitle: "Implement algorithms for the Multi Armed Bandit Problem"
seoDescription: "An article that explains the multi armed bandit problem and implements the algorithms that can be used to solve the problem in python."
datePublished: 2026-08-01T20:23:50.865Z
cuid: cmsatk8xd00000ahq8mue595v
slug: implementing-algorithms-for-the-multi-armed-bandit-problem
cover: https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/1130679f-c9da-4693-84df-8e4bc275e020.jpg
tags: ai, machine-learning, reinforcement-learning, multi-armed-bandit, exploration-exploitation

---

Let's suppose you are faced with a dilemma; there are *k* possible actions to take, and you have only *T* opportunities to take actions. Taking an action returns a reward R from an underlying reward distribution (but of course you do not know the reward distribution for each action). Your ultimate goal is to maximize your rewards

This is the problem framed by the Multi Armed Bandit, the actions are called the arms and when you "pull" an arm you receive a reward according to the reward distribution associated with that arm.

There are various real-world problems that can be viewed as a multi-armed-bandit problem, let us look at a few:

1.  An advertising company wants its users to click the displayed ads as much as possible, there are multiple ads from which to pick from and there is an unknown pick probability attributed to each ad. This is basically a multi armed bandit problem, the ads are the arms, the pick probability is the reward distribution, and the reward is 1 when the user clicks on the add and 0 when they don't.
    
2.  You're playing a game and you are trying to win the game as much as possible, there are multiple strategies for winning the game, each strategy has a particular probability of winning the game (which of course you do not know). This is another MAB problem where the strategies are the arms, winning the game is a reward of 1 and a reward of 0 for losing the game and the probability of winning a game for a given strategy is the reward distribution.
    

So, these are some real-world examples of the MAB problem, and we are here to implement the various algorithms that can be used to solve the MAB problem.

## Exploration vs Exploration

Just thinking about the problems above you can realize that there are basically two things that need to be done. We would need to firstly observe all the available actions (pull all arms) this is to see how "good" the action is, this is Exploration. The next thing that would need to be done would be to actually pull the good "arm" this is where we actually achieve the goal of maximizing reward.

Then another dilemma arises from this realization, we cannot know the best action without exploring (we cannot exploit properly without exploration) and of course we cannot maximize our rewards by simply exploring. To properly maximize rewards, we would have to properly balance exploration and exploitation.

This is actually a recurring problem in Reinforcement Learning and a Multi Armed Bandit is a special case (albeit an overly simplified one) of a Reinforcement Learning Problem. Understanding Exploration-Exploitation trade off here would actually help you understand it in more complex RL systems.

Before we actually implement the algorithms with code, we would have to set up the MAB environment.

## Setting up the Environment

We would need to setup the MAB class, which is a standardization of the agent and how it should work.

```python
## Prepare Environment

from jaxtyping import Float, Array
import numpy as np
from typing import Callable, Union
import matplotlib.pyplot as plt

np.random.seed(316)

## Helper function, would be used eventually
def random_argmax(ary: Array) -> int:
    """Take an argmax and randomize between ties."""
    max_idx = np.flatnonzero(ary == ary.max())
    return np.random.choice(max_idx).item()

class MAB:
    """The Bernoulli multi-armed bandit environment."""

    def __init__(self, means: Float[Array, " K"], T: int):
        """Constructor.

        :param means: the means (success probabilities) of the reward distributions for each arm
        :param T: the time horizon
        """
        # Ensure means are between 0 and 1
        assert np.all((means >= 0) & (means <= 1)), "Means must be between 0 and 1"
        
        self.means = np.asarray(means)
        self.T = T
        self.K = self.means.size
        self.best_arm = random_argmax(self.means)

    def pull(self, k: int) -> int:
        """Pull the `k`-th arm and sample from its (Bernoulli) reward distribution."""
        # A Bernoulli trial: returns 1 with probability self.means[k]
        reward = np.random.rand() < self.means[k]
        return int(reward)

class Agent:
    """Stores the pull history and uses it to decide which arm to pull next.
        
    Since we are working with Bernoulli bandit,
    we can summarize the pull history concisely in a (K, 2) array.
    """

    def __init__(self, K: int, T: int):
        """The MAB agent that decides how to choose an arm given the past history."""
        self.K = K
        self.T = T
        self.rewards = []  # for plotting
        self.choices = []
        self.observations = dict()
        self.history = np.zeros((K, 2), dtype=int)

    def choose_arm(self) -> int:
        """Choose an arm of the MAB. Algorithm-specific."""
        pass

    def count(self) -> int:
        """The number of pulls made. Also the current step index."""
        return len(self.rewards)

    def update_history(self, arm: int, reward: int):
        self.rewards.append(reward)
        self.choices.append(arm)
        if arm in self.observations.keys():
            self.observations[arm] += reward
        else:
            self.observations[arm] = reward
        self.history[arm, reward] += 1
```

A couple of things to note in the code:

1.  The agent is able to take an action
    
2.  The MAB environment is able to give a reward for an action taken
    
3.  The agent is able to track all its choices and the rewards it gets for each choice.
    
4.  There is a reward distribution associated with each arm. In the code we have K arms and their distributions are given by the means attribute for the MAB class.
    
5.  There is an underlying best arm, which is the arm that would give the best reward. In this implementation we basically set the reward distributions for all the arms, but in the real world we do not have access to this information.
    

Now we would need the mab loop, which basically coordinates the MAB agent to choose an arm, pull the arm, receive a reward for pulling the arm and then update its observations and history.

```python
# The MAB loop

def mab_loop(mab: MAB, agent: "Agent") -> int:
    for t in range(mab.T):
        ## Select an action
        arm = agent.choose_arm()  # in 0, ..., K-1
        ## Take the action and receive a reward for the action
        reward = mab.pull(arm)
        ## update history and observations
        agent.update_history(arm, reward)

## Initialize the mab agent
mab = MAB(means=np.array([0.1, 0.8, 0.4]), T=100)
```

After writing the code for the loop, we can initialize the environment we would be working with for this article.

there are 3 available actions:

1.  action 0 would give a reward of 1, 10 percent of the time.
    
2.  action 1 would give a reward of 1, 80 percent of the time.
    
3.  action 2 would give a reward of 1, 40 percent of the time.
    

Now let's move on to the actual algorithms that can be used to solve the MAB problem and how we can implement and evaluate them.

## Pure Exploration

I am sure you have figured out what this algorithm entails. Here the agent randomly selects an action every time. You can already guess that this is not a very good approach. Yeah sure, it is not a very good approach intuitively but how do we prove it. This is where "Regret" comes in.

### Regret

This is a metric for evaluating the performance of MAB algorithms by comparing the reward of the algorithm to the reward that would have been obtained if it had followed an "oracle" (who knows the underlying reward distribution of each action) who always picks the best arm (in our case that is arm 1). It is basically the difference between the algorithms reward and the reward of picking the best action.

Let us implement its calculation

```python
## Introducing Regret, plotting the regret per step and cumulative regret

def regret_per_step(mab: MAB, agent: Agent):
    """Get the difference from the average reward of the optimal arm. The sum of these is the regret."""
    return [mab.means[mab.best_arm] - mab.means[arm] for arm in agent.choices]

def plot_strategy(mab: MAB, agent: Agent, size: int = 40):
    plt.figure(figsize=(4, 2))

    # plot reward and cumulative regret
    plt.plot(np.arange(mab.T), np.cumsum(agent.rewards), label="reward")
    cum_regret = np.cumsum(regret_per_step(mab, agent))
    plt.plot(np.arange(mab.T), cum_regret, label="cumulative regret")

    # draw colored circles for arm choices
    colors = ["red", "green", "blue"]
    color_array = [colors[k] for k in agent.choices]
    plt.scatter(np.arange(mab.T), np.zeros(mab.T), c=color_array, s = size)

    # Add legend entries for each arm
    for i, color in enumerate(colors):
        plt.scatter([], [], c=color, label=f'arm {i}')

    # labels and title
    plt.xlabel("Pull index")
    plt.legend()
    plt.show()
```

The regret per step function calculates the regret at each time step (action opportunity) and then there is code to basically plot the reward over time as well as the regret over time.

Let us implement the fairly simple pure exploration algorithm and then see the regret plot for it.

```python
## Pure Exploration

class PureExploration(Agent):
    def choose_arm(self) -> int:
        """Choose an arm uniformly at random."""
        # np.random.randint(low, high) is exclusive of high, 
        # so this picks from 0, 1, ..., K-1
        choice = np.random.randint(0, self.K)
        return choice


agent = PureExploration(mab.K, mab.T)
mab_loop(mab, agent)
plot_strategy(mab, agent)
```

So here we defined the pure exploration agent, ran the mab loop and then plotted the regret plot for the algorithm.

Here is the direct result from the code

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/a00404d9-0073-411c-80ad-68a1facfcb42.png align="center")

here we can see that the regret just keeps rising, and on the X axis we can see the pure exploration agent in action, it just picks arms at random.

We have successfully evaluated the aproach and it has a pretty linear regret that keeps increasing with time and for the current environment it reaches a little over 30. (high regret is bad btw)

Now let us actually use our history to exploit the best action.

## Pure Greedy

Unlike the Pure Exploration approach, this approach actually exploits the best action, but it does this naively. Every possible action is only taken once, then the action that gave the highest reward is exploited continuously. This is definitely better than the pure exploration approach, but it is still not very good.

Let us see why, first we would implement the algorithm and run the mab loop.

```python
class ExploreThenCommit(Agent):
    def __init__(self, K: int, T: int, N_explore: int):
        super().__init__(K, T)
        self.N_explore = N_explore
        self.explored = 0
        self.window = []  # Initialize a window to track exploration
        
    def choose_arm(self):
        while self.explored < self.N_explore:
            for arm in range(self.K):
                if arm not in self.window:
                    self.window.append(arm)
                    return arm
            self.window = []
            self.explored += 1
        
        return max(self.observations, key=self.observations.get)
    
agent = ExploreThenCommit(mab.K, mab.T, mab.T // 15)
mab_loop(mab, agent)
plot_strategy(mab, agent)
                
                
```

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/5c980579-61d6-44c3-84d3-6225a725e2a1.png align="center")

In this particular run above, the agent actually found the best arm after exploring just once. but remember that we are dealing with probability distributions, so how often would the agent find the best arm?

Running the loop a couple more times, we see that the model would not always get the best arm after exploring once, because an arm with a reward distribution of 0.1 could give a reward of 1 on the first try while the arm with a reward distribution of 0.9 could give a 0 on the first try, if the model uses this initial observation to make choices, then it could end up "exploiting" the wrong arm.

Here is the regret plot after running it a second time:

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/485ca1a8-f50a-4414-9ca3-f66f0d0f76fc.png align="center")

We can see here that the model did not pick the best arm and went with a sub optimal arm instead. In fact, from our environment definitions {action 0 would give a reward of 1, 10 percent of the time, action 1 would give a reward of 1, 80 percent of the time, action 2 would give a reward of 1, 40 percent of the time.}. The agent is actually "exploiting" the worst arm.

We can improve on this algorithm by simply increasing how many times we explore all actions.

## Explore-then-commit

Here the agent would firstly explore all possible actions a fixed number of times and then just like the pure greedy approach, it exploits the best action and does so continuously without ever looking back.

Here are the code implementation and the regret plot for this approach.

```python
class ExploreThenCommit(Agent):
    def __init__(self, K: int, T: int, N_explore: int):
        super().__init__(K, T)
        self.N_explore = N_explore
        self.explored = 0
        self.window = []  # Initialize a window to track exploration
        
    def choose_arm(self):
        while self.explored < self.N_explore:
            for arm in range(self.K):
                if arm not in self.window:
                    self.window.append(arm)
                    return arm
            self.window = []
            self.explored += 1
        
        return max(self.observations, key=self.observations.get)
    
agent = ExploreThenCommit(mab.K, mab.T, mab.T // 15)
mab_loop(mab, agent)
plot_strategy(mab, agent)
                
                
```

Pure Greedy is actually just a special case of explore then commit with the N\_explore set to 1.

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/b767324f-93f8-4955-83fc-da43e216d9c8.png align="center")

The Explore then commit algorithm would definitely perform better than the pure greedy algorithm more often than not. But how long would it take to find the best arm? To properly maximize rewards and minimize regret the agent would have to find a good balance between exploration and exploitation.

ETC splits them into two separate phases, this is not the best we can do, why can't we exploit while we explore?

Having a pure exploration phase leaves valuable rewards on the table. Let us see how we can fuse both the exploration and exploitation phases.

> NB: Even though ETC would find the best arm more times than pure greedy would, it is still not guaranteed that it would find the best arm. Depending on the value of N\_explore the agent may or may not find the best arm. But of course, the larger the N\_explore the more likely the agent is to find the best arm, and also the more times it just randomly selects arms which greatly adds to the regret (Exploration phase leaves valuable rewards on the table). This is also not a very good approach to maximize rewards.

## Epsilon Greedy Approach

The epsilon greedy algorithm fuses both exploration and exploitation phases by randomly deciding whether to Explore or to Exploit at each time step. This is an improvement on the ETC approach, instead of Exploring then Exploiting, we randomly decide whether to explore or exploit.

```python
## Epsilon greedy

class EpsilonGreedy(Agent):
    def __init__(
        self,
        K: int,
        T: int,
        E_array: Float[Array, " T"],
    ):
        super().__init__(K, T)
        self.E_array = E_array

    def choose_arm(self):
        E = self.E_array[self.count()]
        if np.random.rand() < E:
            return np.random.randint(0, self.K)
        else:
            ## estimate the expected reward from each arm relative to how many times it has been pulled
            estimates = [self.observations[arm] / self.history[arm].sum() if self.history[arm].sum() > 0 else 0 for arm in range(self.K)]
            return random_argmax(np.array(estimates))

agent = EpsilonGreedy(mab.K, mab.T, np.full(mab.T, 0.1))
mab_loop(mab, agent)
plot_strategy(mab, agent)
```

Epsilon greedy algorithm

1.  randomly generate a number between 1 and 0
    
2.  check if number is lesser than Epsilon.
    
3.  If true, randomly pick an arm. (Explore)
    
4.  If false, look through observations and pick the arm with the highest average reward so far. (Exploit)
    

Now let us see how much better our agent is doing now that we have fused both the exploration and exploitation phases.

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/c8744821-500e-40a3-9aef-c21e50ccb5fc.png align="center")

The regret plot for Epsilon greedy is better than the one for ETC. BUT we can still do better. There is a problem with the epsilon greedy approach, at some point (after we have taken a fairly large number of actions) exploration could be quite.... unnecessary. The rate at which we Explore stays fixed throughout the loop, but thinking intuitively, after exploring for some time, we should exploit more than we explore because we would be more certain about the estimates of the "quality" of the actions.

The next approach is the first adaptive approach that improves its estimate of the quality of each arm as it makes more observations.

## Upper Confidence Bound

This is the first adaptive approach we would be implementing. Here we continually update an estimate of the quality of an arm, and we also update our certainty of that estimate. This leads to a natural balance between exploration and exploitation where we not only pick arms based on estimates, but also based on our certainty of the estimates.

The empirical mean of the arm is given by:

$$\hat{\mu}_i = \frac{R_i}{N_i}$$

where R is the sum of rewards we have gotten from pulling arm i and N is the number of times arm i has been pulled.

We define our level of certainty of an estimate by using a confidence bound, given by:

$$\sqrt{\frac{\ln\left(\frac{2t}{\delta}\right)}{2N_i}}$$

Where t is the number of times we have pulled an arm (not specific this time, this is basically the number of time steps we have had). The confidence bound basically quantifies our certainty of the estimate of an arm, (the larger the bound the higher the uncertainty) you can infer that as the bound decreases with increase in N (number of times we have pulled an arm) meaning that naturally we are more certain about arms that we have selected a large number of times.

The Upper Confidence bound becomes:

$$\text{UCB}_i = \hat{\mu}_i + \sqrt{\frac{\ln\left(\frac{2t}{\delta}\right)}{2N_i}}$$

Then we select the arm with the highest confidence bound:

$$A_t = \arg\max_i \text{UCB}_i$$

the code implementation of this algorithm is given below:  

```python
## Upper Confidence Bound (UCB)

class UCB(Agent):
    def __init__(self, K: int, T: int, delta: float):
        super().__init__(K, T)
        self.delta = delta
        self.window = [] # Initialize a window to track initial exploration

    def choose_arm(self):
        for arm in range(self.K):
            if arm not in self.window:
                self.window.append(arm)
                return arm
        estimates = [self.observations[arm] / self.history[arm].sum() if self.history[arm].sum() > 0 else 0 for arm in range(self.K)]
        conf_bounds = [estimate + np.sqrt(np.log((2 * len(self.choices))/ self.delta) / (2 * self.history[arm].sum())) if self.history[arm].sum() > 0 else np.inf for arm, estimate in enumerate(estimates)]
        return random_argmax(np.array(conf_bounds))


agent = UCB(mab.K, mab.T, 0.9)
mab_loop(mab, agent)
plot_strategy(mab, agent, size = 20)
```

Intuitively, UCB prioritizes arms where:

1.  estimate\[arm\] is large, the arm’s corresponding sample mean is high, and we’d choose it for exploitation, and
    
2.  np.sqrt(np.log((2 \* len(self.choices))/ self.delta)/(2 \* self.history\[arm\].sum())) is large or self.history\[arm\].sum() is small and we’re still uncertain about the arm, and we’d choose it for exploration.
    

estimate\[arm\] represents the current estimate of the quality of that arm (the estimate of its reward distribution).

while: np.sqrt(np.log((2 \* len(self.choices))/ self.delta)/(2 \* self.history\[arm\].sum())) represents our certainty of that estimate, it is the confidence bound.

This represents how certain we are about the estimate of the arms quality. This naturally balances exploration and exploitation by prioritizing arms that either have a large estimate for its quality (exploitation) or arms that we aren't very sure about its quality (exploration).

Below is the regret plot for UCB.

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/07e562f3-f06b-4783-b9d2-c3c659558dfb.png align="center")

## Thompson Sampling

This approach uses probability to make estimates of the quality of the actions. We take a Bayesian approach where we treat them as random variables from some prior distribution. Then, upon pulling an arm and observing a reward, we can simply *condition* on this observation to exactly describe the posterior distribution over the parameters. This fully describes the information we gain about the parameters from observing the reward.

In other words, we sample each arm proportionally to how likely we think it is to be optimal, given the observations so far. This strikes a good exploration-exploitation tradeoff: we explore more for arms that we’re less certain about and exploit more for arms that we’re more certain about. Thompson sampling is a simple yet powerful algorithm that achieves state-of-the-art performance in many settings. (from [here](https://rlbook.adzc.ai/bandits.html#introduction))

Each arm is assumed to have an unknown success probability:

$$\theta_i \sim \text{Beta}(1,1)$$

After observing S successes and F failures, the posterior becomes:

$$\theta_i \mid \mathcal{D} \sim \text{Beta}(1+S_i,;1+F_i)$$

At each round, sample from each posterior:

$$\tilde{\theta}_i \sim \text{Beta}(1+S_i,;1+F_i)$$

Then choose the arm with the highest sampled value:  

$$A_t = \arg\max_i \tilde{\theta}_i$$

Here is the code implementation:

```python
## Thompson Sampling 

class ThompsonSampling(Agent):
    def choose_arm(self) -> int:
        """
        Sample from the posterior Beta distribution for each arm and 
        select the arm with the highest sample.
        """
        # Beta(1 + successes, 1 + failures)
        samples = [
            np.random.beta(1 + self.history[arm, 1], 1 + self.history[arm, 0])
            for arm in range(self.K)
        ]
        return random_argmax(np.array(samples))

agent = ThompsonSampling(mab.K, mab.T)
mab_loop(mab, agent)
plot_strategy(mab, agent, size = 20)
```

Here is its regret plot:

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/719f9932-ba5a-46c6-afbf-43f58a86115b.png align="center")

## Comparing all the algorithms

At some point after the ETC algorithm, it became a little unclear which algorithm actually performed best.

To get a more descriptive and clearer evaluation of the various algorithms, I ran all of them 10 times on a much more complex environment, with 10 arms each having their own reward distribution, assigned randomly. I ran the loop for 2000 timesteps then I averaged the regret for all the algorithms at each time step. Then i plotted the average regret against the timesteps and voila.

We can clearly see that Thompson Sampling is the Best here (Remember that the lower the regret the better the model). Epsilon Greedy and UCB are close seconds, then the rest ETC, Pure greedy then Pure Exploration.

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/7ab30af6-8f2d-4524-a06b-6444c2f81c10.png align="center")

Understanding these various algorithms for solving MAB problems would equip you with the intuition to understand the Exploration-Exploitation trade off in more complex RL environments.

## References

1.  The notebook with all the code used in this project can be found [here](https://github.com/Badaszz/RL_playground/tree/main/Multi-armed-bandit)
    
2.  [Multi-Armed Bandits – An Introduction to Reinforcement Learning](https://rlbook.adzc.ai/bandits.html#introduction)
    
3.  Reinforcement Learning: An Introduction Richard S. Sutton and Andrew G. Barto. Chapter 2
    
4.  The UCB algorithm explanation was gotten from this YouTube video [here](https://youtu.be/s6UHInwoqb0?si=7ZfQI5sUEIL3IqnE)
    
5.  This YouTube video explains some of the algorithms, especially Thompson sampling. [here](https://youtu.be/gNQXBNgO8zo?si=8iAMPoBF_t1QYqSp)