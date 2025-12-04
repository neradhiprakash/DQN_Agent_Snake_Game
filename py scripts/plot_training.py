import csv
import matplotlib.pyplot as plt

games = []
scores = []
records = []
means = []
epsilons = []

# READ CSV
with open("training_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        games.append(int(row["game"]))
        scores.append(int(row["score"]))
        records.append(int(row["record"]))
        means.append(float(row["mean_score"]))
        epsilons.append(float(row["epsilon"]))

# COMPUTE exploration/exploitation from epsilon
rand_pcts = [e / 2.0 for e in epsilons]            # % random moves
model_pcts = [100.0 - (e / 2.0) for e in epsilons] # % model moves

# PLOT 1 – SCORE / MEAN / RECORD
plt.figure(figsize=(12,6))
plt.plot(games, scores, label="Score")
plt.plot(games, means, label="Mean Score")
plt.plot(games, records, label="Record")
plt.xlabel("Game")
plt.ylabel("Score")
plt.title("Training Progress: Score / Mean / Record")
plt.legend()
plt.grid()
plt.savefig("1_scores_plot.png")
plt.close()

# PLOT 2 – EPSILON DECAY
plt.figure(figsize=(12,6))
plt.plot(games, epsilons, label="Epsilon")
plt.xlabel("Game")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay Over Training")
plt.legend()
plt.grid()
plt.savefig("2_epsilon_plot.png")
plt.close()

# PLOT 3 – EXPLORATION vs EXPLOITATION
plt.figure(figsize=(12,6))
plt.plot(games, rand_pcts, label="% Random Moves")
plt.plot(games, model_pcts, label="% Model Moves")
plt.xlabel("Game")
plt.ylabel("Percentage")
plt.title("Exploration vs Exploitation During Training")
plt.legend()
plt.grid()
plt.savefig("3_exploration_plot.png")
plt.close()

print("Plots saved successfully!")
