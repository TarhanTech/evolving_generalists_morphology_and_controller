import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    envs = make_vec_env("CartPole-v1", n_envs=4)
    envs.reset()

    print("OBSERVATION SPACE")
    print(f"Observation Space Shape: {envs.observation_space.shape}")
    print(f"Sample observation: {envs.observation_space.sample()}")

    print("ACTION SPACE")
    print(f"Action Space Shape: {envs.action_space.shape}")
    print(f"Action Space Sample: {envs.action_space.sample()}")

    # model = PPO("MlpPolicy", envs, verbose=1)
    # model.learn(total_timesteps=25000)
    # model.save("my_ppo_cartpole")
    # del model

    model = PPO.load("my_ppo_cartpole")

    eval_env: CartPoleEnv = gym.make("CartPole-v1", render_mode="human")
    # mean_reward, std_reward = evaluate_policy(model, eval_env, render=True, n_eval_episodes=5, deterministic=True, warn=False)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    
    obs, _ = eval_env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Total Reward: {total_reward}")
            break


if __name__ == "__main__": main()