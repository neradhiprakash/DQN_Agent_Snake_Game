import torch
from snake_game import SnakeGame
from dqn_agent import Agent


def play():
    # Create game and agent
    game = SnakeGame()
    agent = Agent()

    # Load the trained model weights
    agent.model.load_state_dict(torch.load("best_model.pth"))

    # no random moves
    agent.n_games = 1000

    # Main loop: let the trained agent play forever
    while True:
        # Get current state from the game
        state = agent.get_state(game)

        # Get action from trained model
        action = agent.get_action(state)

        # Apply action to the game
        reward, done, score = game.play_step(action)

        # If game ends, show score and restart
        if done:
            print("Game Over | Score:", score)
            game.reset()


# Run the play
if __name__ == "__main__":
    play()
