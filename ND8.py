import numpy as np
import time

params = {
    "discount": 0.9,
    "living_reward": -0.5,
    "reward": 10,
    "defeat": -10,
    "boardCol": 5,
    "boardRow": 4,  
}

board = [
    [1, 1, 1, 1, 1],
    [1, 0, 1, 1, 10], 
    [1, 0, 0, 1, -10],
    [1, 1, 1, 1, 1],
]

actions = ["up", "down", "left", "right"]

def get_next_state(state, action):
    row, col = state
    if action == "up": row -= 1
    elif action == "down": row += 1
    elif action == "left": col -= 1
    elif action == "right": col += 1
    return (row, col)

def is_valid_state(state):
    row, col = state
    return 0 <= row < params["boardRow"] and 0 <= col < params["boardCol"] and board[row][col] != -10

class MDP:
    def __init__(self, board, config, start):
        self.board = board
        self.discount = config["discount"]
        self.living_reward = config["living_reward"]
        self.board_col = config["boardCol"]
        self.board_row = config["boardRow"]
        self.start = start
        self.rewards = np.zeros((self.board_row, self.board_col))
        self.value_grid = np.zeros((self.board_row, self.board_col))
        self.policy_grid = np.empty((self.board_row, self.board_col), dtype=object)

        for row in range(self.board_row):
            for col in range(self.board_col):
                if self.board[row][col] == 10:
                    self.rewards[row, col] = params["reward"]
                elif self.board[row][col] == -10:
                    self.rewards[row, col] = params["defeat"]
                else:
                    self.rewards[row, col] = self.living_reward

    def get_actions(self, state):
        row, col = state
        actions = []
        if row > 0 and self.board[row - 1][col] != 0:
            actions.append((-1, 0))
        if row < self.board_row - 1 and self.board[row + 1][col] != 0:
            actions.append((1, 0))
        if col > 0 and self.board[row][col - 1] != 0:
            actions.append((0, -1))
        if col < self.board_col - 1 and self.board[row][col + 1] != 0:
            actions.append((0, 1))
        return actions

    def transition(self, state, action):
        row, col = state
        new_row = max(0, min(self.board_row - 1, row + action[0]))
        new_col = max(0, min(self.board_col - 1, col + action[1]))
        return (new_row, new_col)

    def value_iteration(self):
        for iteration in range(1, 1000):
            new_value_grid = np.zeros((self.board_row, self.board_col))
            for row in range(self.board_row):
                for col in range(self.board_col):
                    if self.board[row][col] != 0:
                        best_value = float('-inf')
                        best_action = None
                        for action in self.get_actions((row, col)):
                            next_state = self.transition((row, col), action)
                            r = self.rewards[next_state[0], next_state[1]]
                            next_value = r + self.discount * self.value_grid[next_state[0], next_state[1]]
                            if next_value > best_value:
                                best_value = next_value
                                best_action = action
                        new_value_grid[row, col] = best_value
                        self.policy_grid[row, col] = best_action
            if np.array_equal(new_value_grid, self.value_grid):
                break
            self.value_grid = new_value_grid

    def print_value_matrix(self):
        for row in range(self.board_row):
            for col in range(self.board_col):
                if (row, col) == self.start:
                    print("S ", end="")
                elif self.board[row][col] == 10:
                    print("X ", end="")
                elif self.board[row][col] == -10:
                    print("L ", end="")
                else:
                    value = f"{self.value_grid[row][col]:.2f}"
                    spacing = " " * (6 - len(value))
                    print(f"({value}){spacing}", end="")
            print()

    def print_optimal_path(self):
        current_state = self.start
        while True:
            row, col = current_state
            action = self.policy_grid[row, col]
            print(f"({row}, {col})")
            if self.board[row][col] == 10:
                print("Victory!")
                break
            elif self.board[row][col] == -10:
                print("Landed on a defeat...")
                break
            current_state = self.transition(current_state, action)

#----------------------- #2 Skatinamojo mokymo algoritmas -----------------------
def reinforcement_learning(epsilon=0.1, alpha=0.1, gamma=0.9, max_iterations=1000, epsilon_decay=0.99, min_epsilon=0.01, minimal_changes=0.01):
    Q = np.zeros((params["boardRow"], params["boardCol"], len(actions)))
    prev_Q = np.copy(Q)

    for iteration in range(max_iterations):
        state = (0,3)

        while True:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                action = actions[np.argmax(Q[state[0], state[1]])]

            next_state = get_next_state(state, action)

            if not is_valid_state(next_state):
                continue

            reward = params["living_reward"]
            if board[next_state[0]][next_state[1]] == params["reward"]:
                reward = params["reward"]
            elif board[next_state[0]][next_state[1]] == params["defeat"]:
                reward = params["defeat"]

            Q[state[0], state[1], actions.index(action)] = (1 - alpha) * Q[state[0], state[1], actions.index(action)] + \
                alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1]]))

            state = next_state

            if state == (1, 4) or state == (2, 4):
                break

        if np.all(np.abs(Q - prev_Q) < minimal_changes):
            print(f"Minimal changes (< 0.01) at iteration {iteration + 1} so learning will be stopped.")
            break

        prev_Q = np.copy(Q)

    return Q

def reinforcement_path(Q):
    current_state = (0,3)
    optimal_path = []
    
    while True:
        row, col = current_state
        action_idx = np.argmax(Q[row, col])
        action = actions[action_idx]
        optimal_path.append((row, col))
        if board[row][col] == 10:
            print("Optimal path:")
            for state in optimal_path:
                print(state)
            print("Victory!")
            break
        elif board[row][col] == -10:
            print("Optimal path:")
            for state in optimal_path:
                print(state)
            print("Landed on a defeat...")
            break
        current_state = get_next_state(current_state, action)

#----------------------- #3 uzd. Modelio pagristas laiko skaiciavimas -----------------------
start = time.time()
mdp = MDP(board, params, start=(0, 3))
mdp.value_iteration()
print("Model-based learning:")
mdp.print_value_matrix()
print("Optimal path:")
mdp.print_optimal_path()
end = time.time()
print("Time taken:", end - start, "seconds")

#----------------------- #3 uzd. Skatinamojo mokymo laiko skaiciavimas -----------------------
start = time.time()
print("\nReinforcement learning:")
Q = reinforcement_learning()
end = time.time()
reinforcement_path(Q)
print("Time taken:", end - start, "seconds")
