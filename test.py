from learner import LearnBrain
import numpy as np
# Word/part-of-speech acquisition experiments
avgs = []
stds = []
areas = ["PHON", "VISUAL", "NOUN"]
decrease_cat = []
heal_cat = []
decrease_jump = []
heal_jump = []
trial = 0
# calculate mean and standard deviation
for i in range(3):
  plasticity = 0.05
  LEX_k = 100
  steps = 30
  percent_neurons_destroyed = 0.4
  cur_results = {"LEX_k": LEX_k, "steps": steps, "percent_neurons_destroted": percent_neurons_destroyed}
  brain = LearnBrain(plasticity, LEX_k=LEX_k)
  # Training the brain
  brain.train_simple(steps)
  cur_results["cat_res_pre_stroke"] = brain.test_word_nums_only("CAT")
  cur_results["jump_res_pre_stroke"] = brain.test_word_nums_only("JUMP")
  # Inducing a stroke
  brain.stroke(percent_neurons_destroyed=percent_neurons_destroyed, areas=areas)
  cur_results["cat_res_post_stroke"] = brain.test_word_nums_only("CAT")
  cur_results["jump_res_post_stroke"] = brain.test_word_nums_only("JUMP")
  decrease_cat.append(cur_results["cat_res_pre_stroke"]["noun_overlap"] - cur_results["cat_res_post_stroke"]["noun_overlap"])
  print(cur_results["cat_res_pre_stroke"]["noun_overlap"], cur_results["cat_res_post_stroke"]["noun_overlap"])
  decrease_jump.append(cur_results["jump_res_pre_stroke"]["noun_overlap"] - cur_results["jump_res_post_stroke"]["verb_overlap"])

  # 'Healing' the brain
  brain.train_simple(steps)
  cur_results["cat_res_healed"] = brain.test_word_nums_only("CAT")
  cur_results["jump_res_healed"] = brain.test_word_nums_only("JUMP")
  heal_cat.append(cur_results["cat_res_post_stroke"]["noun_overlap"] - cur_results["cat_res_healed"]["noun_overlap"])
  heal_jump.append(cur_results["jump_res_post_stroke"]["verb_overlap"] - cur_results["jump_res_healed"]["verb_overlap"])

  # results.append(cur_results)

print(np.mean(decrease_cat))
print(decrease_cat)