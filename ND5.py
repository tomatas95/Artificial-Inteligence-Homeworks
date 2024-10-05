import numpy as np

params = {
    "living_reward": -0.5,
    "reward": 10,
    "defeat": -10,
    "boardCol": 5,
    "boardRow": 4
}

board = [
    [1, 1, 1, 1, 1],
    [1, 0, 1, 1, 10], 
    [1, 0, 0, 1, -10],
    [1, 1, 1, 1, 1],
]

def get_next_state(state, action):
    row, col = state
    if action == "up":
        row -= 1
    elif action == "down":
        row += 1
    elif action == "left":
        col -= 1
    elif action == "right":
        col += 1

    return (row, col)

def is_valid_state(state):
    row, col = state
    return 0 <= row < params["boardRow"] and 0 <= col < params["boardCol"] and board[row][col] != -10

def model_based_incentive_learning():
    values = np.zeros((params["boardRow"], params["boardCol"]))
    actions = ["up", "down", "left", "right"]
    
    optimal_actions = np.empty((params["boardRow"], params["boardCol"]), dtype=object)

    delta = float('inf')
    iteration = 0
    max_iterations = 1000
    while delta > 0.01 and iteration < max_iterations:
        for row in range(params["boardRow"]):
            for col in range(params["boardCol"]):
                state = (row, col)
                if not is_valid_state(state):
                    continue

                max_value = float("-inf")
                best_action = None
                for action in actions:
                    next_state = get_next_state(state, action)
                    if not is_valid_state(next_state):
                        continue
                    reward = board[next_state[0]][next_state[1]]
                    next_value = values[next_state[0]][next_state[1]]
                    expected_value = reward + params["living_reward"] + next_value
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = action
                delta = max(delta, abs(max_value - values[row][col]))
                values[row][col] = max_value
                optimal_actions[row][col] = best_action
        
        iteration += 1

    return values, optimal_actions

def print_optimal_path(optimal_actions):
    current_state = (3,3)
    while True:
        row, col = current_state
        action = optimal_actions[row][col]
        print(f"({row}, {col})")
        if board[row][col] == 10:
            print("Reached finish wooo!")
            break
        elif board[row][col] == -10:
            print("Landed on a defeat...")
            break
        current_state = get_next_state(current_state, action)


optimal_values, optimal_actions = model_based_incentive_learning()

print("\nOptimal path:")
print_optimal_path(optimal_actions)
