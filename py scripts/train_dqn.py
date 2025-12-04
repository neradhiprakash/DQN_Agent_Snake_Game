import csv

from snake_game import SnakeGame
from dqn_agent import Agent

def train():
    game = SnakeGame()
    agent = Agent()

    scores = []
    epsilons = []
    record = 0

    #create CSV file
    with open("training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
        "game",
        "score",
        "record",
        "mean_score",
        "epsilon",
        "random_moves",
        "model_moves",
        "total_moves",
        "random_pct",
        "model_pct",
    ])
    while True:

        # 1. get old state
        state_old = agent.get_state(game)

        # 2. get action from agent
        action = agent.get_action(state_old)

        # 3. perform action in game
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        # 4. train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # 5. store in memory
        agent.remember(state_old, action, reward, state_new, done)

        # 6. ONLY NOW we check if done
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            scores.append(score)
            epsilons.append(agent.epsilon)

            if score > record:
                record = score
                agent.model.save("best_model.pth")

            mean_score = sum(scores) / len(scores)

            #append one row to CSV
            with open("training_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    agent.n_games,
                    score,
                    record,
                    mean_score,
                    agent.epsilon,
                ])

            print(
                f"Game {agent.n_games:4d} | "
                f"Score {score:2d} | "
                f"Record {record:2d} | "
                f"Mean {mean_score:5.2f} | "
                f"Epsilon {agent.epsilon:5.2f}"
            )

            #show how many moves were random vs model based
            total_moves = agent.random_moves + agent.model_moves
            if total_moves > 0:
                rand_pct = 100 * agent.random_moves / total_moves
                model_pct = 100 * agent.model_moves / total_moves
                print(
                    f"  Moves-> Random: {agent.random_moves} ({rand_pct:.1f}%), "
                    f"Model: {agent.model_moves} ({model_pct:.1f}%)"
                )
            # resetting counters for next game
            agent.random_moves = 0
            agent.model_moves = 0

if __name__ == "__main__":
    train()
