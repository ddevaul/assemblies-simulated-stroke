from learner import LearnBrain
import numpy as np
import pickle as pkl 
# Word/part-of-speech acquisition experiments
def stroke_diff_severities_overlap_pre_stroke(areas, stroke_severities, plasticity, simple_train_steps, recovery_steps, pre_stroke_percentage):
  total_sv_vals = []
  total_sv_assembly_vals = []
  for severity in stroke_severities:
    words = [("CAT", "noun"), ("DOG", "noun"), ("JUMP", "verb"), ("RUN", "verb")]
    brain = LearnBrain(plasticity, LEX_k=100)
    total_results = []
    total_results_assemblies = []
    # brain.stroke(percent_neurons_destroyed=pre_stroke_percentage, areas=["PHON"])
    brain.train_simple(simple_train_steps)
    running = []
    running_assemblies = []
    for word in words:
      res = brain.test_word_nums_only(word[0])
      running.append(res[f"{word[1]}_overlap"] / len(res[f"{word[1]}_winners"]))
    
    total_results.append(np.mean(running))

    for noun_idx in range(4):
      out = brain.testIndexedWord(noun_idx)
      out = out if out else -1
      if out == noun_idx:
        running_assemblies.append(1)
      else:
        running_assemblies.append(0)
    total_results_assemblies.append(np.mean(running_assemblies))

    brain.stroke(percent_neurons_destroyed=severity, areas=areas)

    for _ in range(recovery_steps):
      brain.train_simple(1)
      running = []
      for word in words:
        res = brain.test_word_nums_only(word[0])
        running.append(res[f"{word[1]}_overlap"] / len(res[f"{word[1]}_winners"]))
      total_results.append(np.mean(running))

      for noun_idx in range(4):
        out = brain.testIndexedWord(noun_idx)
        out = out if out else -1
        if out == noun_idx:
          running_assemblies.append(1)
        else:
          running_assemblies.append(0)
      total_results_assemblies.append(np.mean(running_assemblies))


    total_sv_vals.append(total_results)
    total_sv_assembly_vals.append(total_results_assemblies)
  return total_sv_vals, total_sv_assembly_vals
  
areas = ["NOUN", "VERB"]
stroke_severities = [0.1, 0.2, 0.3]
plasticity = 0.05
simple_train_steps = 30
recovery_steps = 10
pre_stroke_percentage = 0.5

results1, results2 = stroke_diff_severities_overlap_pre_stroke(areas, stroke_severities, plasticity, simple_train_steps, recovery_steps, pre_stroke_percentage)
results = {"res1" : results1, "res2": results2}
with open("combined_results.pkl", "wb") as f:
  pkl.dump(results, f)