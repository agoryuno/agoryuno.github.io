---
layout: post
mathjax: true
title:  "Multi-Armed Bandits with Thompson Sampling: a toy example from finance"
date:   2022-07-27
---

We use a Thompson Sampling Multi-Armed Bandit (MAB TS) to explore solutions to a simple financial game, which I've first encountered in the [Doxod company's Telegram channel](https://t.me/dohod/11500) (in Russian).

The complete code for this example is available at [https://github.com/agoryuno/mab_ts_example](https://github.com/agoryuno/mab_ts_example)

# Rules of the game

The game is formulated as follows: we have a 1000 shares in 3 traded funds, named "green", "blue" and "red". At the beginning of the game shares in all three funds cost the same, so for simplicity's sake we assume the starting share price to be equal to 1 for all funds. The game is played for 20 rounds and at the end of each round each of the funds' share prices can grow at a rate selected from the following possibilities:

    "green" : {0.8, 0.9, 1.1, 1.1, 1.2, 1.4},
    "blue"  : {0.95, 1, 1, 1, 1, 1.1},
    "red"   : {0.05, 0.2, 1, 3, 3, 3}

Looking at the growth rate possibilities we can tell that the funds have different expected rates of return and corresponding risks, with the "red" fund being the riskiest and having the highest maximum return, and "blue" - the least risky with the lowest max return rate.

## Game objective

The objective of the game is obviously to get the most profit out of our initial investment. To add a bit more realism we'll discount the returns, arbitrarily setting the discount rate to be 0.05. It is needed to note that due to the purely random nature of returns and complete lack of correlations, there's no "correct solution" to this game, per se. However, some solutions will be demonstrably better than others "on average", therefore our aim is to choose the best general approach out of several options for constructing an investment portfolio given this setting. 

Note specifically that we are not looking for "the best" or "the optimal" approach - the algoritm can only choose between the options we propose but can't evolve new options. Keep in mind that this is a classification algorithm at heart.

# Multi-Armed Bandits

There's a lot of information online about Multi-Armed Bandits in general, and Multi-Armed Bandits with Thompson Sampling in particular. Multi-Armed Bandits are essentially a family of classification algorithms, with Thompson Sampling being the Bayesian inference flavor. There's no practical need to use MAB with our game but we'll do it anyway just to explore the algorithm itself.

## The reward function

The central part of MAB estimation is the reward function. This is some function which given an action (or an arbitrarily complicated sequence of actions) returns a numeric reward. In probably the most popular practical setting for MAB - modelling user response to ads - this reward is binary, with 1 for clicking on an ad and 0 - ignoring it. In our case it'll be a real number. For the purposes of simplifying sampling we need to keep the scale of the reward fairly tight - the game can generate wildly different returns with maximums exceeding the means by a factor of thousand. Therefore, instead of discounted profit we'll use the natural logarithm of the ratio of the fully discounted value of our portfolio at the end of the game to its starting value (1000 by problem definition) :

$$r_a = \log{ \frac{v_a}{V (1+d)^T}},    (1)$$

where $$V$$ - initially invested capital ($$V=1000$$), $$d$$ - discount rate ($$d=0.05$$), $$T$$ - number of rounds ($$T=20$$), $$v_a$$ - value of the portfolio in the end of the game. 

## The policy

The value $$v_a$$ in (1) is the result of implementing a policy $$a$$. This is an action or a complex of actions which, when executed, yields a certain result, represented by $$v_a$$. In our case a policy is a specific investment strategy that is followed for the number of periods $$T$$ and yields value $$v_a$$ in the end. We'll define a number of such strategies, presented below.



```python
import math
import random
from abc import ABC, abstractmethod

STRATEGIES = [
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1./3, 1./3, 1./3],
        [0.3, 0.6, 0.1],
        [0.1, 0.8, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.3, 0.3],
        [0.3, 0.3, 0.4],
        [0.3, 0.1, 0.6]
    ]

assert all([math.isclose(sum(row),1) for row in STRATEGIES])
```

In the code above, each row of the `STRATEGIES` matrix contains shares of, respectively, "green", "blue", and "red" funds in the initial portfolio.  Thus, the first 3 strategies represent completely undiversified portfolios, the 4th is fully balanced, the next 3 - conservative portfolios with the least risky fund contributing at least half of the total value, the last 3 - riskier portfolios, with 60% of the last one contributed by the super risky "red" fund.

Additionally, for all portolios other than the first 3 we formulate rebalancing strategies, aimed at keeping the portfolio close to its initial makeup as the fund prices change throughout the game. 

To reinforce the idea that a strategy (we'll just keep using this term, implying a policy in standard MAB parlance) can be as simple as a single function or method that returns a reward value, we introduce an abstract class Strategy with a single required method `__call__()`. This is a special method which will be executed when an instance of that class is called as a function.


```python
from abc import ABC, abstractmethod

class Strategy(ABC):
    
    @abstractmethod
    def __call__(self):
        ...

```

And use that class for the implementation of the actual investment strategies. This is probably the most complicated part of the entire model, since the MAB TS algorithm itself is very straightforward. Lets take the Strategy in parts, starting with the `__init__()` method.


```python
class StrategyInvest(Strategy):

    def __init__(self, shares, 
                 init_stock=INIT_STOCK, 
                 num_rounds=NUM_ROUNDS,
                 discount=DISCOUNT_RATE,
                 outcomes=OUTCOMES):
        
        """
        shares - a dictionary with fund names as keys and fund shares as 
                 values
        init_stock - the starting total number of shares in all funds
        num_rounds - the number of rounds the game will be played for
        discount - the discount rate
        outcomes - a dictionary with all possible outcomes (growth rates)
                   for each fund, where fund names are keys and outcomes are
                   lists of rates. Outcome frequencies are represented by 
                   including the same outcome more than once (order is not 
                   important), e.g. [1, 1.1, 1.1, 0.9] means that probability 
                   of 1 and 0.9 is 0.25 and the probability of 1.1 is 0.5
        """
        
        self.shares = shares
        assert all(k in self.shares for k in ("green", "blue", "red"))
        
        self.portfolio = {k : init_stock*v for k,v in self.shares.items()}
        self.prices = {fund : [1] for fund in self.portfolio.keys() }
        
        self.outcomes = outcomes
        self.num_rounds = num_rounds
        
        self.__curr_round = 0
        self.discount = discount
        self.init_capital = sum([v for v in self.get_values().values()])

```

The default settings are in global constants, which should be above the `Strategy` class in the code file.


```python
INIT_STOCK = 1000.
NUM_ROUNDS = 20
DISCOUNT_RATE = 0.05

OUTCOMES = {
    "green" : [0.8, 0.9, 1.1, 1.1, 1.2, 1.4],
    "blue" : [0.95, 1, 1, 1, 1, 1.1],
    "red" : [0.05, 0.2, 1, 3, 3, 3]}
```

Next are the various "utility" methods.


```python
    def _choose_outcome(self, fund):
        """ Randomly chooses an outcome for a given fund """
        return random.choice(self.outcomes[fund])
    
    def _get_portfolio_return(self):
        """ 
        Returns the ratio of the discounted portfolio value 
        to initial capital value
        """
        return self._get_total_value() / \
            (self.init_capital*(1+self.discount)**self.__curr_round)
    
    def _get_total_value(self):
        """ 
        Returns the portfolio's total value given current 
        fund share prices 
        """
        return sum([value for value in self.get_values().values()])
    
    def get_values(self):
        """
        Returns a dictionary with fund names as keys and current 
        fund values as values
        """
        return {fund : self.prices[fund][-1]*stock 
                    for fund, stock in self.portfolio.items()}
    
    def value_shares(self):
        """
        Returns a dictionary with fund names as keys and current 
        shares of the corresponding funds in the total portfolio 
        value as values
        """
        values = self.get_values()
        total_value = np.sum(list(values.values()))
        return {fund : value / total_value for fund, value in values.items()}
       
```

And the `__call__()` method required by the abstract class.


```python
    def __call__(self):
        for i in range(self.num_rounds):
            for fund in self.portfolio.keys():
                self.prices[fund].append(
                    self.prices[fund][-1]*self._choose_outcome(fund)
                )
            self.__curr_round = i
        return self._get_portfolio_return()
```

Putting the above three code blocks together gives us the basic `StrategyInvest` class without rebalancing. To add rebalancing of fund shares we subclass `StrategyInvest`, adding a `rebalancing()` method and overriding the `__call__()` method. 


```python
class StrategyInvestRebalancing(StrategyInvest):
    
    def __call__(self):
        ret = super().__call__()
        self.rebalance()
        return ret

    def rebalance(self):
        values = self.get_values()
        total_value = self._get_total_value()
        value_shares = self.value_shares()
        share_diffs = {fund : share - self.shares[fund] 
                        for fund, share in value_shares.items()}

        cash = 0
        for fund, diff in share_diffs.items():
            if diff >= 0.1:
                cutback = (diff - 0.1)*total_value
                cash += cutback
                self.portfolio[fund] -= cutback/self.prices[fund][-1]
        if cash > 0:
            buyins = []
            for fund, diff in share_diffs.items():
                if diff < 0:
                    buyins.append ( 
                        (fund, -1*(total_value*diff)/self.prices[fund][-1],
                                    -1*(total_value*diff), diff) 
                    )
            if len(buyins) == 1:
                buyin = buyins[0]
                quant_buy = cash/self.prices[buyin[0]][-1]
                self.portfolio[buyin[0]] += quant_buy
                return
            total_diff = np.sum([b[-1]*-1 for b in buyins])
            diff_shares = [(b[0],(b[-1]*-1)/total_diff) for b in buyins]
            for ds in diff_shares:
                quant_buy = (cash*ds[1])/self.prices[ds[0]][-1]
                self.portfolio[ds[0]] += quant_buy
```

We won't go into the details of the `rebalance()` implementation. Suffice it only to say that it tries its best to shift the excess value from the funds with portfolio shares exceeding initial by more than 10 percentage points into funds with shrunk shares. It does this after every round.

The code we have so far is available as a single file at [https://github.com/agoryuno/mab_ts_example/blob/main/mab_ts1.py](https://github.com/agoryuno/mab_ts_example/blob/main/mab_ts1.py)

To make sure everything works you can run a couple of strategies:


```python
from mab_ts1 import StrategyInvest, StrategyInvestRebalancing, STRATEGIES

strat1 = StrategyInvest(
    {fund : share for fund, share in 
         zip(("green", "blue", "red"),STRATEGIES[0])}
)
strat1()

strat2 = StrategyInvestRebalancing(
    {fund : share for fund, share in 
         zip(("green", "blue", "red"),STRATEGIES[4])}
)
strat2()
```

One not immediately obvious problem with this approach to building strategies is that further along, once we start to actually use it, it'll produce runaway simulations. This is because the `Strategy` objects, once created, maintain their state between subsequent executions of their `__call__()` method. There is a simple solution for this, which we'll address when we get to using the strategies.

# The algorithm

The MAB algorithm assign a strategy to an arm of the bandit and playing that arm executes the strategy, yielding a reward. We'll implement the bandit as two base classes: `Arm` and `Bandit`, where multiple instances of `Arm` will be "attached" to a single `Bandit`.

## The arms

We'll do the same thing for the `Arm` and `Bandit` as we did for the `Strategy` above: make base classes with the most fundamental functionality and extend them with specific implementations by subclassing. 

Code for the base `Arm` class follows:



```python
class Arm:

    def __init__(self, play):
        """
        play - a callable object that returns a reward value
        """
        self._play = play
        self.rewards = []
    
    @property
    def times_played(self):
        """
        Number of times this arm's been played
        """
        return len(self.rewards)
        
    @property
    def mean_reward(self):
        """
        The mean reward received over all plays of this arm
        """
        return np.mean(self.rewards)
    
    def update(self):
        raise NotImplementedError
    
    def __call__(self):
        reward = self._play()
        self.rewards.append(reward)
        self.update()
```

The base `Arm` class is initialized with a single argument: `play`. This is a callable object which when called returns a reward value. Our `Strategy` class - not coincidentally - satisfies those requirements.

Instances of the `Arm` class are themselves callable, as is evident from the presence of the `__call__()` method. When called, the instance records a reward and "updates" itself. The actual implementation of the `update()` method is left to the inheriting subclasses to define and is the substance of Thomson Sampling.

## Thompson Sampling

"Thompson Sampling", as [described by Wikipedia](https://en.wikipedia.org/wiki/Thompson_sampling), "consists of choosing the action that maximizes the expected reward with respect to a randomly drawn belief". What that means in plain English is that when faced with a choice between two or more options ("Arms" in our context) we choose that which we *believe* will yield the highest reward.

Being more concrete, our *belief* in an Arm's reward is described by a certain distribution of the values of reward, specified by selected values for the given distribution's parameters. In other words, before starting to repeatedly play a Bandit, we choose a distribution function which - in our *belief* - more or less accurately describes how the true (produced by the "world") values of the Arms' rewards are distributed. We then choose (usually heuristically) the starting values for the distributions' parameters, and then sample a random value from the distribution for each Arm and pull the Arm which receives the highest value.

As the game progresses, and for each pull of an Arm we receive an actual reward, in a perfectly Bayesian spirit we update the values of the distributions' parameters (for each Arm separately - each Arm maintains its own set of parameters). That process of repeated updates to the initial *belief* results in the distributions "narrowing" down towards the true mean rewards.

As the distribution for each Arm "narrows", both extremely small and extremely large values become increasingly *less likely* to be drawn, and that (specifically the decreasing likelihood of drawing extremely large values) decreases the probability of an Arm being chosen if the mean of its distribution is smaller than that of the other Arms. Therefore, the Arm that has the highest mean becomes *more likely* to be drawn as the game progresses.

On the other hand, if some Arm (pun intended) is picked more rarely than others - its distribution will remain "wider" than the rest, thus the probability of drawing an extremely large value for that Arm is relatively higher, which increases its chances of being chosen.

That mechanism of counter-acting probabilities is how Thompson Sampling solves the "explore-exploit" dilemma that is central to all types of reinforcement learning: it focuses on using (exploiting) the Arm with the highest expected mean reward but leaves space for the other Arms to be chosen every once in a while (exploring).

## Updating an Arm

The `update()` method of the `Arm` class is then key to making Thompson Sampling work. To implement this method we first need to choose the distribution that we *believe* more or less accurately represents how the actual rewards are distributed. However, our choice needs to be guided by expedience alongside with accuracy. By this I mean that whatever distribution we choose for our likelihood function, which tells us how likely a certain reward is given the distribution's parameters, it must have a readily available "conjugate" prior distribution, which tells us how likely the parameter values are before the actual rewards are factored in.

In other words, the parameters of the likelihood function's distribution are sampled from the prior distribution and the hypotetical reward is then sampled from the likelihood function's distribution with the sampled parameters. The importance of the two distribution's being "conjugate" lies in the fact that update equations of parameters of such a pair of distributions can be derived algebraically, making the update process fast and simple.

What's more, people have already found solutions for quite a few pairs of conjugate distributions and Wikipedia offers a comprehensive [table of the most commonly used combinations](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions).

For our problem we'll use the Normal-NormalGamma pair of distributions. Although, since we are using log-reward, as opposed to straight reward, it is actually LogNormal-NormalGamma.

We will sample our rewards from the normal distribution with parameters $$\mu$$ and $$\frac{1} {\lambda \tau}$$, with $$\tau$$ being sampled from the gamma distribution with parameters $$\alpha$$ and $$\beta$$.

The update equation for $$\mu$$:

$$
\mu' = \frac {\lambda  \mu + n  \bar {x}} {\lambda + n}
$$

where $$\mu'$$ is the updated value of $$\mu$$, $$n$$ - the number of samples we've already taken (equiv. the number of times the Arm's been played), $$\bar {x}$$ - the mean of the actual rewards received.

For $$\beta$$:

$$
\beta' = \beta + \frac{1}{2} \sum_{i=0}^{n-1}{(x_i - \bar{x})^2} + \frac {n \lambda} {\lambda + n} \frac {(\bar{x} - \mu)^2} {2}
$$

where $$\beta'$$ is the updated value of $$\beta$$, $$x_i$$ is an i-th factual reward value.

And the update equations for $$\alpha$$ and $$\lambda$$ are much simpler:

$$
\alpha' = \alpha + \frac {n}{2}
$$

$$
\lambda' = \lambda + n
$$

where $$\alpha'$$ and $$\lambda'$$ are the updated values.

With these equations we can proceed to construct the practical implementation of an Arm. Once again we begin with the `__init__()` method.


```python
class ArmNormalGamma(Arm):

    def __init__(self, play, alpha=1, beta=1, 
                 mu=0, lmd=0, **kwargs):
        super().__init__(play, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.lmd = lmd
```

Next is the `update()` method, which simply codifies the update equations above:


```python
    def update(self):
        n = self.times_played
        m = self.mean_reward
        self.mu = (self.lmd*self.mu + n*m)/ \
                  (self.lmd + n)
        self.lmd +=  n
        self.alpha += n/2.
        rewards = np.array(self.rewards)
        self.beta += 0.5*np.sum((rewards-m)**2) + \
                (n*self.lmd)/(self.lmd+n) * (0.5 * (m - self.mu)**2)
```

And the method to sample the hypothetical reward used in the Arm selection process:


```python
    def sample(self):
        tau = gamma.rvs(self.alpha, 1./self.beta)
        return norm.rvs(self.mu, np.sqrt(1/(self.lmd*tau)) )
```

Note that above we sample $$\tau$$ from the Gamma distribution. $$\tau$$ is the "precision" parameter for the formulation of the Normal distribution density functions alternative to the more often cited standard deviation one. Scipy uses the "traditional" formulation, but $$\frac {1} {\lambda \tau}$$ just so happens to be equal to variance, and since variance is standard deviation squared, $$sqrt {\frac {1}{\lambda \tau}}$$ gives us the standard deviation we can plug into Scipy's normal distribution sampling function.

We also need to override the base `Arm` class' `__call__()` method, since we need to take the log of reward


```python
    def __call__(self):
        reward = self._play()
        if reward == 0:
            reward = 1e-9
        reward = np.log(reward)
        self.rewards.append(reward)
        self.update()
```

In the `__call__()` method above we account for the (tiny) possibility of reward being equal to 0, which would produce negative infinity as a result.

Now is a good time to come back to the problem with using `Strategy` objects that was mentioned before. Unfortunately we cannot simply do something like this:


```python
strat = StrategyInvest(
    {fund : share for fund, share in 
         zip(("green", "blue", "red"),STRATEGIES[0])}
)

arm = ArmNormalGamma(strat)
for i in range(1000):
    arm()
```

If we do that, then results of subsequent `strat()` calls will cause the `Strategy.prices` dictionary to continually accumulate records, which will amount to the game running beyond the set limit of 20 rounds, potentially allowing our portfolio to grow to infinity.

To avoid this happening we can create the strategy object every time the arm is played, like this:


```python
strat = lambda : StrategyInvest(
    {fund : share for fund, share in 
         zip(("green", "blue", "red"),STRATEGIES[0])}
)()

arm = ArmNormalGamma(strat)
for i in range(1000):
    arm()
```

Now, instead of passing to the arm the `Strategy` object itself, we are passing it a lambda function which, when called from inside the `Arm.__call__()` method will return the result of creating and immediately calling a `Strategy` object.

That finalizes our formulation of the bandit's arm. 

The complete code we have so far is at [https://github.com/agoryuno/mab_ts_example/blob/main/mab_ts2.py](https://github.com/agoryuno/mab_ts_example/blob/main/mab_ts2.py)

## Testing the arms

Now that we have the strategies and arms fully working, we can run a short and simple test to see how (and if) they work in tandem. Using the 'mab_ts2.py', run the following code:


```python
from matplotlib import pyplot as plt

from mab_ts2 import (STRATEGIES, ArmNormalGamma, 
                     StrategyInvestRebalancing)
strat = lambda : StrategyInvestRebalancing(
    {fund : share for fund, share in 
         zip(("green", "blue", "red"),STRATEGIES[3])}
)()

arm = ArmNormalGamma(strat)

fig, axes = plt.subplots(2,2)
fig.set_size_inches(10., 10.)

def _(axes, row, col, N=100):
    mus, means = [], []
    for i in range(N):
        arm()
        mus.append(arm.mu)
        means.append(arm.mean_reward)


    axes[row][col].plot(range(N), mus, label="mu")
    axes[row][col].plot(range(N), means, label="mean")
    
_(axes, 0, 0)
_(axes, 0, 1)
_(axes, 1, 0)
_(axes, 1, 1)
```

That code will produce a figure similar to the one below. In the graphs the orange jagged line represents the actual mean reward produced by the arm plays, while the smooth blue line depicts the values of the $$\mu$$ paramater - the hypothetical mean.

![figure1.png](/assets/images/figure1.png)

Note that due to the random nature of the game no two runs of the simulation will produce the same result. However, in all 4 graphs the values of the actual and hypothetical means are close together, and in three cases out of four the lines actually appear to converge. The latter is merely an illusion, but this convergence represents the tendency of the orange line to return to the blue one. A tendency which is better revealed in the next figure, which represents 20 000 pulls on each of the four arms.

![figure1_1.png](/assets/images/figure1_1.png)

In the above figure we can see four different types of convergence. In the top left graph the convergence between the hypothetical and factual means appears absolute, although that is yet another illusion, caused by the bigger scale of the vertical axis due to the abnormally high and low means at the start of the run. In the top right graph the orange line repeatedly touches the blue, in the bottom left - it crosses the blue from both above and below, and in the bottom right - stays below but close to the blue. 

In all four cases the hypothetical mean tends to roughly the same value - around 0.4. This is important because although the arms in this experiment are different, they all play the same strategy, so the means should be close despite the random nature of the game.

Based on this brief examination we can conclude that our strategies work together with the arms and produce the expected results, leaving us free to move further to construct the actual Bandit.

# The bandit

The Bandit is the simplest part of the entire model. All it needs to do is "assemble" a collection of arms, provide a method for choosing the next arm to be played and the usual `__call__()` method to run the simulation.


```python
class Bandit:
    
    def __init__(self, arms, T, warmup=1):
        """
        arms - an iterable of Arm objects
        T - the total number of times to play the Bandit
        warmup - the number of times to play each Arm in the
                 warmup cycle
        """
        
        self.arms = [arm for arm in arms]
        self.T = T
        self.warmup = warmup
    
    def _choose_arm(self):
        samples = [arm.sample() for arm in self.arms]
        idx = np.argmax(samples)
        return self.arms[idx]
    
    def __call__(self):
        for i in range(self.warmup):
            for arm in self.arms:
                arm()
        
        for i in range(self.T):
            arm = self._choose_arm()
            arm()
            
```

All we have left to do is build a function to construct a list of Arms from the STRATEGIES matrix we've defined all the way at the very beginning of this longwinded manuscript.


```python
def make_arms(smatrix=STRATEGIES):
    strats = []
    for row in smatrix:
        strats.append(
            lambda: StrategyInvest(
                {fund : share 
                  for fund, share in 
                     zip(("green", "blue", "red"),row)})()
        )
    for row in smatrix[3:]:
        strats.append(
            lambda: StrategyInvestRebalancing(
                {fund : share 
                  for fund, share in 
                     zip(("green", "blue", "red"),row)})()
        )
    return [ArmNormalGamma(strat) for strat in strats]
```

This function takes a matrix of strategies in the same format as the `STRATEGIES` we've defined before (and it takes the `STRATEGIES` object as its only argument's default value) and returns a list of `ArmNormalGamma` objects that can be passed to a `Bandit`.

The entire complete code is at [https://github.com/agoryuno/mab_ts_example/blob/main/mab_ts_final.py](https://github.com/agoryuno/mab_ts_example/blob/main/mab_ts_final.py)

# Results and analysis

Now we can proceed to run our Bandit and perhaps get some idea about which types of strategies it thinks are better than others for playing our game.
