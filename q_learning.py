import numpy as np
import random
import os
import matplotlib.pyplot as plt
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
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
    values = []
    for a in q.actions:
        values.append(q.get(s, a))
    return max(values)
 
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
    def get(self, s, a):
        # Your code here
        return self.models[self.actions.index(a)].predict(tf.convert_to_tensor(s), verbose=0)
    def update(self, data, lr, epochs=1):
        # Your code here
        X_all = [np.zeros((1, len(self.states))) for _ in range(len(self.actions))]
        Y_all = [np.zeros((1, 1)) for _ in range(len(self.actions))]
        a_idxs = set()
        for each in data:
            s, a, t = each
            
            a_idxs.add(self.actions.index(a))
            if not np.any(X_all[self.actions.index(a)]):
                Y_all[self.actions.index(a)] = np.array([[np.squeeze(t)]])
                X_all[self.actions.index(a)] = s
            else:
                Y_all[self.actions.index(a)] = np.append(Y_all[self.actions.index(a)], np.array([[np.squeeze(t)]]), 0)
                X_all[self.actions.index(a)] = np.append(X_all[self.actions.index(a)], s, 0)
            
        for i in list(a_idxs):
            self.models[i].fit(tf.convert_to_tensor(np.array(X_all[i])), tf.convert_to_tensor(np.array(Y_all[i])), epochs=epochs, verbose=0)
        
        return
   
def rnd_choice(s):
    actions = [i for i in range(N_HOLES) if s[0, 0, i] != 0] 
    if not actions:
        return 0
    return random.choice(actions)
    
def sim_episode(mdp, episode_length, policy, draw=False):
    episode = []
    reward = 0
    s = mdp.init_state()
    all_states = [s]
    for i in range(int(episode_length)):
        mdp.board.print_board()
        a = policy(s)
        if mdp.terminal():
            (r, s_prime) = mdp.sim_transition(a)
            reward += r
            break
        (r, s_prime) = mdp.sim_transition(a)
        reward += r
        
        (r, s) = mdp.sim_transition(rnd_choice(s_prime))
        all_states.append(s)
    #animation = animate(all_states, mdp.n, episode_length) if draw else None
    return reward, episode, 0
    
    
def test_learn_play(num_layers = 10, num_units = 100,
                    eps = 0.5, iters = 10000,
                    num_episodes = 10, episode_length = 100):
      
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
    
    qf = Q_learn(game, q, iters=iters, interactive_fn=None)
    
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