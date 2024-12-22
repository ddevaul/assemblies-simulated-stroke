import pickle as pkl 
import matplotlib.pyplot as plt
with open("combined_results.pkl", "rb") as f:
  data = pkl.load(f)

results1 = data["res1"]
results2 = data["res2"]
colors = ["red", "blue", "green"]
stroke_severities = [0.1, 0.2, 0.3]
areas = ["NOUN", "VERB"]
scaled_xvals = [i-1 for i in range(len(results1[0]))]
for i in range(len(results1)):
  plt.plot(scaled_xvals , results1[i], color=colors[i], label=f"{stroke_severities[i]}")
  plt.plot(scaled_xvals , results2[i], color=colors[i], label=f"{stroke_severities[i]}", marker="*")
  plt.title(f"Stroke in {areas}")
  plt.ylabel("average overlap / winners")
  plt.xlabel("recovery steps")
plt.legend(title="percent neurons destroyed")
plt.show()