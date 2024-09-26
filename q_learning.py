import numpy as np
import random
import os 
import matplotlib.pyplot as plt
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
tf.config.optimizer.set_jit(True)
import keras
from keras.models import Sequential # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.layers import Dense # type: ignore
from game import N_HOLES, N_STONES
from game import Game

        
# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    # Your code here (COPY FROM HW9)
    actions = [a for a in q.actions if s[0, 0, a] != 0]
    maximum = -np.inf
    for a in actions:
        v = q.get(s, a)
        if v > maximum:
            maximum = v
    return maximum
 
# Given a state, return the action that is greedy with reespect to the
# current definition of the q function
def greedy(q, s):
    # Your code here (COPY FROM HW9)
    actions = [a for a in q.actions if s[0, 0, a] != 0]
    action_to_choose = None
    maximum = -np.inf
    for a in actions:
        if q.get(s, a) > maximum:
            maximum = q.get(s, a)
            action_to_choose = a
    if not action_to_choose:
        return random.choice(actions)
    return action_to_choose

def epsilon_greedy(q, s, eps = 0.5):
    if random.random() < eps:  # True with prob eps, random action
        actions = [a for a in q.actions if s[0, 0, a] != 0]
        if not actions:
            return random.choice(q.actions)
        return random.choice(actions)
    else:
        return greedy(q, s)
  
def Q_learn(mdp, q, lr=.1, iters=100, eps = 0.5, interactive_fn=None):
    # Your code here
    s = mdp.init_state()
    for i in range(iters): 
        a = epsilon_greedy(q, s, eps)
        curr_player = mdp.player
        r, s_prime = mdp.sim_transition(a)
        if mdp.terminal():
            future_val = 0
        elif curr_player == mdp.player:
            future_val = value(q, s_prime)
        else:
            future_val = -value(q, s_prime)
        q.update([(s, a, (r + mdp.discount_factor * future_val))], lr)
        s = s_prime
        if interactive_fn: interactive_fn(q, i)
    return q

def Q_learn_batch(mdp, q, lr=.1, iters=100, eps=0.5,
                  episode_length=100, n_episodes=1,
                  interactive_fn=None):
    # Your code here
    all_experience = []
    for i in range(iters):
        for _ in range(n_episodes):
            episode = sim_episode(mdp, episode_length, lambda s: epsilon_greedy(q, s, eps))
            all_experience.extend(episode)
        all_q_targets = []
        for experience in all_experience:
            s, a, r, s_prime, player = experience
            if player == None:
                future_val = 0
            elif player == 0:
                future_val = value(q, s_prime)
            else:
                future_val = -value(q, s_prime)
            all_q_targets.append((s, a, r + mdp.discount_factor*future_val))
        q.update(all_q_targets, lr)
    return q
            

  
def make_nn(state_dim, num_hidden_layers, num_units):
    model = Sequential()
    model.add(keras.Input(shape=(1, 14)))
    for i in range(num_hidden_layers-1):
        model.add(Dense(num_units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model      
        
class NNQ:
    def __init__(self, states, actions, num_layers, num_units, epochs=1):
        self.actions = actions
        self.states = states
        self.epochs = epochs
        state_dim = self.states.size
        self.models = [make_nn(state_dim, num_layers, num_units) for _ in range(len(actions))]              # Your code here
        self.predictors = []
        for i in range(6):
            @tf.function(reduce_retracing=True)
            def f(x):
                return self.models[i](x)
            self.predictors.append(f)
    def get(self, s, a):
        # Your code here
        return self.predictors[a](s)
    def update(self, data, lr, epochs=1):
        # Your code here
        
        X_all = [[] for _ in range(6)]
        Y_all = [[] for _ in range(6)]
        for each in data:
            s, a, t = each
            X_all[a].append(s)
            Y_all[a].append(np.array([[np.squeeze(t)]]))
            
        for i in range(6):
            self.models[i].fit(X_all[i], Y_all[i], epochs=epochs)
            #self.models[i].train_on_batch(np.array(X_all[i]), np.array(Y_all[i]))
        
        return
   
def rnd_choice(s):
    actions = [i for i in range(N_HOLES) if s[0, 0, i] != 0] 
    if not actions:
        return 0
    return random.choice(actions)
    
def sim_episode(mdp, episode_length, policy, draw=False):
    episode = []
    s = mdp.init_state()
    for i in range(int(episode_length)):
        a = policy(s)
        print(i)
        if mdp.terminal():
            (r, s_prime) = mdp.sim_transition(a)
            episode.append((s, a, r, None, None))
            break
        (r, s_prime) = mdp.sim_transition(a)
        episode.append((s, a, r, s_prime, mdp.player))
        s = s_prime

    return episode
    
    
def test_learn_play(num_layers = 2, num_units = 50,
                    eps = 0.5, iters = 50,
                    num_episodes = 100, episode_length = 100):
      
    iters_per_value = 1 if iters <= 10 else int(iters / 10.0)
    scores = []
    def interact(q, iter=0):
        x = 0
        if iter % iters_per_value == 0:
            x += 0.001
            scores.append((iter, evaluate(game, num_episodes, episode_length,
                                          lambda s: greedy(q, s))[0]))
            #print('score', scores[-1], flush=True)
    game = Game()
    q = NNQ(game.states, game.actions, num_layers, num_units,
                epochs=1)
    
    #qf = Q_learn(game, q, iters=iters, interactive_fn=None)
    qf = Q_learn_batch(game, q, iters=iters)
    
    if scores:
        # Plot learning curve
        plot_points(np.array([s[0] for s in scores]),
                    np.array([s[1] for s in scores]))
        
    for i in range(num_episodes):
        game.play_ai(lambda s: greedy(qf, s)) 
        #reward, _, animation = sim_episode(game, episode_length,
        #                        lambda s: greedy(qf, s), draw=False)
        #print('Reward', reward)
    #return animation


def tidy_plot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
    # plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def plot_points(x, y, ax = None, clear = False, 
                  xmin = None, xmax = None, ymin = None, ymax = None,
                  style = 'or-'):

    if ax is None:
        if xmin == None: xmin = np.min(x) - 0.5
        if xmax == None: xmax = np.max(x) + 0.5
        if ymin == None: ymin = np.min(y) - 0.5
        if ymax == None: ymax = np.max(y) + 0.5
        ax = tidy_plot(xmin, xmax, ymin, ymax)

        x_range = xmax - xmin; y_range = ymax - ymin
        if .1 < x_range / y_range < 10:
            plt.axis('equal')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x, y, style, markeredgewidth=0.0)
    # Seems to occasionally mess up the limits
    # ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, which='both')
    plt.show()
    return ax

def evaluate(mdp, n_episodes, episode_length, policy):
    score = 0
    length = 0
    for i in range(n_episodes):
        # Accumulate the episode rewards
        r, e, _ = sim_episode(mdp, episode_length, policy)
        score += r
        length += len(e)
        # print('    ', r, len(e))
    return score/n_episodes, length/n_episodes


if __name__ == "__main__":
    test_learn_play()