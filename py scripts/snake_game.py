import pygame
import random
from enum import Enum
import sys

# --- Game constants ---
WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20
SNAKE_SPEED = 20  # frames per second

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (200, 0, 0)
GREEN = (0, 200, 0)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGame:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h

        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake Game (AI)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 25)

        self.reset()

    def reset(self):
        # initial direction
        self.direction = Direction.RIGHT

        # snake head
        x = self.w // 2
        y = self.h // 2
        self.head = [x, y]
        self.snake = [self.head[:],
                      [x - BLOCK_SIZE, y],
                      [x - 2 * BLOCK_SIZE, y]]

        # score
        self.score = 0

        # food
        self._place_food()

        # for future use (e.g. limits), but not needed yet
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randrange(0, self.w, BLOCK_SIZE)
            y = random.randrange(0, self.h, BLOCK_SIZE)
            if [x, y] not in self.snake:
                self.food = [x, y]
                break

    def play_step(self, action):
        """
        action is one of [0, 1, 2]:
            0 -> go straight
            1 -> turn right
            2 -> turn left
        """

        self.frame_iteration += 1

        # 1. Handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 2. Move the snake
        self._move(action)
        self.snake.insert(0, self.head[:])

        # 3. Check if game over (collision)
        reward = -0.1  #penalty for each move
        game_over = False

        if self._is_collision(self.head):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # too many moves - stuck - end episode
        if self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = +10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SNAKE_SPEED)

        # 6. Return game status
        return reward, game_over, self.score


    def _is_collision(self, pt):
        x, y = pt

        # hit wall
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return True

        # hit itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # draw snake
        for segment in self.snake:
            pygame.draw.rect(
                self.display,
                GREEN,
                pygame.Rect(segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE),
            )

        # draw food
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE),
        )

        # score text
        text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, (10, 10))

        pygame.display.flip()

    def _move(self, action):
        """
        action: 0 = straight, 1 = right, 2 = left
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:  # no change
            new_dir = clock_wise[idx]
        elif action == 1:  # right turn
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # action == 2, left turn
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = [x, y]


# quick test: random AI playing
if __name__ == "__main__":
    game = SnakeGame()

    while True:
        # random action: 0 (straight), 1 (right), 2 (left)
        action = random.randint(0, 2)
        reward, done, score = game.play_step(action)

        if done:
            print("Final Score:", score)
            pygame.time.delay(1500)
            game.reset()
