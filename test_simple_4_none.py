from learner_more_words import LearnBrain
import numpy as np
# Word/part-of-speech acquisition experiments
def stroke_diff_severities_many_words_pre_stroke(areas, num_nouns, num_verbs, stroke_severities, plasticity, simple_train_steps, recovery_steps, pre_stroke_percentage):
  total_sv_vals = []
  for severity in stroke_severities:
    brain = LearnBrain(plasticity, LEX_k=100, num_nouns=num_nouns, num_verbs=num_verbs)
    total_results = []
    # brain.stroke(percent_neurons_destroyed=pre_stroke_percentage, areas=areas)
    print("training")
    brain.train(simple_train_steps)
    running = []
    nouns = ["DOG", "CAT", "TREE", "PENCIL", "SCISSORS", "COMB", "FLOWER", "TOOTHBRUSH", "BROOM", "MUSHROOM",  "CAMEL", "BENCH", "SNAIL", "DART", "GLOBE", "WREATH", "BEAVER", "ACORN", "STILTS", "DOMINOES", "CACTUS", "HARP",  "KNOCKER", "STETHOSCOPE", "UNICORN", "FUNNEL", "COMPASS", "TRIPOD",  "SCROLL", "TRELLIS", "PALETTE", "ABACUS", "JUMP", "RUN"]
    # for noun_idx in range(len(nouns)):
    for noun_idx in range(num_nouns + num_verbs):
      out = brain.testIndexedWord(noun_idx)
      out = out if out else -1
      if out == noun_idx:
        running.append(1)
      else:
        running.append(0)

    total_results.append(np.mean(running))

    brain.stroke(percent_neurons_destroyed=severity, areas=areas)

    for _ in range(recovery_steps):
      brain.train(1)
      running = []
      for noun_idx in range(num_nouns + num_verbs):
        out = brain.testIndexedWord(noun_idx)
        out = out if out else -1
        if out == noun_idx:
          running.append(1)
        else:
          running.append(0)
      total_results.append(np.mean(running))
    total_sv_vals.append(total_results)
  return total_sv_vals
  

import matplotlib.pyplot as plt 
areas = ["NOUN", "VERB", "MOTOR", "VISUAL"]
stroke_severities = [0.1, 0.5, 0.9]
plasticity = 0.05 
simple_train_steps = 30
recovery_steps = 10
pre_stroke_percentage = 0.85
num_nouns = 4
num_verbs = 4
total_vals = stroke_diff_severities_many_words_pre_stroke(areas, num_nouns, num_verbs, stroke_severities, plasticity, simple_train_steps, recovery_steps, pre_stroke_percentage)
print(total_vals)
scaled_xvals = [i-1 for i in range(len(total_vals[0]))]
for i in range(len(total_vals)):
  plt.plot(scaled_xvals , total_vals[i], label=f"{stroke_severities[i]}")
  plt.title(f"Stroke in {areas}")
  plt.ylabel("average overlap / winners")
  plt.xlabel("recovery steps")
plt.legend(title="percent neurons destroyed")
plt.show()

import pickle as pkl 
with open("stroke_diff_severities_many_words_none.pkl", "wb") as f:
  pkl.dump(total_vals, f)