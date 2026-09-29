from collections import deque

import gymnasium as gym
import minigrid  # registers the environments

ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
N_EPISODES = 50

# Actions: 0 = turn left, 1 = turn right, 2 = forward, 3 = pick up, 6 = done
LEFT, RIGHT, FORWARD, PICKUP, DONE = 0, 1, 2, 3, 6
DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up


def find_target(env):
    u = env.unwrapped
    for x in range(u.width):
        for y in range(u.height):
            cell = u.grid.get(x, y)
            if cell and cell.type == u.targetType and cell.color == u.targetColor:
                return (x, y)
    return None


def plan(env):
    """Breadth-first search over (x, y, direction) to the target."""
    u = env.unwrapped
    target = find_target(env)
    x, y = map(int, u.agent_pos)
    start = (x, y, int(u.agent_dir))
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        (x, y, d), path = queue.popleft()
        dx, dy = DIR_TO_VEC[d]
        if (x + dx, y + dy) == target:
            return path + [PICKUP]
        for action in (LEFT, RIGHT, FORWARD):
            if action == LEFT:
                nxt = (x, y, (d - 1) % 4)
            elif action == RIGHT:
                nxt = (x, y, (d + 1) % 4)
            else:
                if u.grid.get(x + dx, y + dy) is not None:
                    continue  # blocked by a wall or object
                nxt = (x + dx, y + dy, d)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [action]))
    return None


def random_agent(env, seed):
    env.action_space.seed(seed)
    return lambda obs: env.action_space.sample()


def oracle_agent(env, seed):
    actions = iter(plan(env) or [])
    return lambda obs: next(actions, DONE)


def run_episode(make_agent, seed):
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=seed)
    act = make_agent(env, seed)
    total_reward, steps = 0.0, 0
    while True:
        obs, reward, terminated, truncated, _ = env.step(act(obs))
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    env.close()
    return total_reward > 0, steps


def evaluate(name, make_agent):
    results = [run_episode(make_agent, seed) for seed in range(N_EPISODES)]
    successes = [steps for ok, steps in results if ok]
    rate = 100 * len(successes) / N_EPISODES
    avg_steps = sum(successes) / len(successes) if successes else float("nan")
    return f"| {name} | {rate:.0f}% | {avg_steps:.1f} |"


if __name__ == "__main__":
    print(f"Environment: {ENV_ID}, {N_EPISODES} episodes (seeds 0-{N_EPISODES - 1})\n")
    print("| Agent | Success rate | Avg steps (successes) |")
    print("|---|---|---|")
    print(evaluate("Random", random_agent))
    print(evaluate("Oracle planner", oracle_agent))
