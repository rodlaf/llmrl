import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from tqdm import trange
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gymnasium as gym
from llamagym import Agent
import random


class CartPoleAgent(Agent):
    def format_prompt(self, observation: gym.core.ObsType) -> str:
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return f"""Balance the pole by moving the cart LEFT or RIGHT.
        cart_pos={cart_pos:.2f}, cart_vel={cart_vel:.2f}, pole_angle={pole_angle:.2f}, pole_vel={pole_vel:.2f}
        """

    def extract_action(self, response: str) -> gym.core.ActType:
        print(f"Response: {response}")
        
        response_upper = response.upper()
        left_pos = response_upper.find("LEFT")
        right_pos = response_upper.find("RIGHT")
        
        if left_pos != -1 and right_pos != -1:
            action = 0 if left_pos < right_pos else 1
        elif left_pos != -1:
            action = 0
        elif right_pos != -1:
            action = 1
        else:
            action = random.choice([0, 1])
            
        print(f"Action: {'LEFT' if action == 0 else 'RIGHT'}\n")
        return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name")
    parser.add_argument("--env", default="CartPole-v1", help="Gym environment")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max new tokens")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for returns")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32"], help="Model dtype")
    parser.add_argument("--project", default="llamagym", help="Wandb project")
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    hyperparams = {
        "model_name": args.model,
        "env": args.env,
        "batch_size": args.batch_size,
        "episodes": args.episodes,
        "max_new_tokens": args.max_tokens,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "do_sample": False,
        "device": args.device,
        "dtype": args.dtype,
    }
    
    wandb.init(project=args.project, config=hyperparams)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
        token=os.environ.get("HF_TOKEN"),
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    agent = CartPoleAgent(
        model, 
        tokenizer, 
        args.device,
        generate_config={"max_new_tokens": args.max_tokens, "do_sample": False},
        training_config={
            "batch_size": args.batch_size, 
            "learning_rate": args.learning_rate, 
            "gamma": args.gamma
        },
    )
    
    env = gym.make(args.env)

    for episode in trange(args.episodes):
        observation, _ = env.reset()
        done = False
        step_count = 0

        while not done:
            action = agent.act(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            agent.assign_reward(reward)
            done = terminated or truncated
            step_count += 1

        # Calculate total return before terminating episode
        total_return = sum(agent.current_episode_rewards)
        train_stats = agent.terminate_episode()
        wandb.log({
            "episode": episode,
            "episode_length": step_count,
            "total_return": total_return,
            **train_stats
        })
