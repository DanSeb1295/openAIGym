import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten

from statistics import mean, median
from collections import Counter
from tqdm import tqdm

LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 300
score_requirement = 50
games = 10000
n_classes = 2
n_nodes = [64, 128, 256]
max_layers = 4
pop_size = 5
latest_gen = 0
max_gen = 30
fittest = None
average_score = 0
fittest_ave_score = 0
optimisers = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
losses = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error",
         "squared_hinge", "hinge", "categorical_hinge", "logcosh", "categorical_crossentropy", "sparse_categorical_crossentropy",
         "binary_crossentropy", "kullback_leibler_divergence", "poisson", "cosine_proximity"]

def initial_population():
   training_data = []
   scores = []
   accepted_scores = []
   for i in tqdm(range(games)):
       score = 0
       game_memory = []
       prev_observation = []
       for _ in range(goal_steps):
           action = random.randrange(0, 2)
           observation, reward, done, info = env.step(action)
           # if _ == 0:
           #     print("observation:", observation)
           if len(prev_observation) > 0:
               game_memory.append([prev_observation, action])
           prev_observation = observation
           score += reward
           if done: break

       if score >= score_requirement:
           accepted_scores.append(score)
           for data in game_memory:
               if data[1] == 0:
                   move = [1, 0]
               elif data[1] == 1:
                   move = [0, 1]
               training_data.append([data[0],move])

       env.reset()
       scores.append(score)

   print('Average accepted score:', mean(accepted_scores))
   print('Median score for accepted scores:', median(accepted_scores))
   print(Counter(accepted_scores))
   return training_data

def generate_random_nn(optimisers, losses):
   optimiser = random.choice(optimisers)
   losses = random.choice(losses)
   # print(optimiser)
   # print(losses)
   model = Sequential()
   model.add(Dense(random.choice(n_nodes), input_dim=4))
   model.add(Dense(random.choice(n_nodes), activation='relu'))

   model.add(Dense(n_classes, activation='sigmoid'))
   model.compile(loss=losses,
                 optimizer=optimiser,
                 metrics=['accuracy'])
   return model

def generate_nn(fittest, training_data):
   model = improve_model(fittest, training_data)
   return model

def improve_model(fittest, training_data):
   X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
   Y = np.array([i[1] for i in training_data])
   fittest.fit(X, Y, epochs=random.randint(1,3))
   return fittest

def generate_population(input_size,max_gen,latest_gen, training_data, fittest=None):
   species = []
   if fittest == None:
       for _ in range(5):
           species.append(generate_random_nn(optimisers, losses))
   else:
       for _ in range(int(3)):
           species.append(generate_nn(fittest, training_data))
       for _ in range(int(2)):
           species.append(generate_random_nn(optimisers, losses))
           for i in species[3:]:
               i = improve_model(i, training_data)
   return species

def play(species):
   choices = []
   scores = []
   all_characters = []
   game_memory = []
   for i in range(len(species)):
       score = 0
       prev_observation = []
       env.reset()
       for _ in range(goal_steps):
           env.render()
           if len(prev_observation) == 0:
               action = random.randrange(0, 2)
           else:
               action = np.argmax(species[i].predict(prev_observation.reshape(-1, len(prev_observation)))[0])

           choices.append(action)

           observation, reward, done, info = env.step(action)

           prev_observation = observation
           game_memory.append([observation, action])
           score += reward
           if done:
               break
       all_characters.append(game_memory)
       env.reset()
       scores.append(score)

   # for _ in range(goal_steps):
   #     env.render()
   #     observation, reward, done, info = env.step(all_characters[np.argmax(scores)][_][1])
   #     if done:
   #         env.reset()
   #         break

   print(scores)
   print('Average Score:', sum(scores) / len(scores))
   print('Max Score:', np.max(scores), "Species:", np.argmax(scores))
   #print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
   return np.argmax(scores), sum(scores) / len(scores)

def run(fittest):
   print("------------------------------------- R U N N I N G   F I T T E S T -------------------------------------")
   choices = []
   scores = []
   all_characters = []
   game_memory = []
   for i in range(100):
       score = 0
       prev_observation = []
       env.reset()
       for _ in range(goal_steps):
           env.render()
           if len(prev_observation) == 0:
               action = random.randrange(0, 2)
           else:
               action = np.argmax(fittest.predict(prev_observation.reshape(-1, len(prev_observation)))[0])

           choices.append(action)

           observation, reward, done, info = env.step(action)

           prev_observation = observation
           game_memory.append([observation, action])
           score += reward
           if done:
               break
       all_characters.append(game_memory)
       env.reset()
       scores.append(score)

   print('Average Score:', sum(scores) / len(scores))
   return sum(scores) / len(scores)

training_data = initial_population()
X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
input_size = len(X[0])
while fittest_ave_score < 195:
   while latest_gen < max_gen and average_score<195:
       species = generate_population(input_size, max_gen, latest_gen, training_data, fittest)
       print("Generation:", latest_gen)
       fittest, average_score = play(species)
       fittest = species[fittest]
       latest_gen+=1
   fittest_ave_score = run(fittest)
