import os
from tqdm import trange
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gymnasium as gym
from llamagym import Agent


class CartPoleAgent(Agent):
    def get_system_prompt(self) -> str:
        return "You control a cart balancing a pole. Respond with 0 (LEFT) or 1 (RIGHT) to keep the pole upright."

    def format_observation(self, observation: gym.core.ObsType) -> str:
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return f"Cart pos={cart_pos:.2f}, Pole angle={pole_angle:.2f}"

    def extract_action(self, response: str) -> gym.core.ActType:
        return 0 if "0" in response else 1


if __name__ == "__main__":
    device = "cuda"
    model_name = "google/gemma-2-2b-it"
    
    hyperparams = {
        "model_name": model_name,
        "env": "CartPole-v1",
        "batch_size": 8,
        "episodes": 1000,
        "max_new_tokens": 5,
        "do_sample": False,
    }
    
    wandb.init(project=os.environ.get("WANDB_PROJECT"), config=hyperparams)
    
    # Load model WITHOUT LoRA - test if base model works first
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float16,
        token=os.environ.get("HF_TOKEN"),
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    agent = CartPoleAgent(
        model, 
        tokenizer, 
        device,
        {"max_new_tokens": hyperparams["max_new_tokens"], "do_sample": hyperparams["do_sample"]},
        {"batch_size": hyperparams["batch_size"]},
    )
    
    env = gym.make(hyperparams["env"])

    for episode in trange(hyperparams["episodes"]):
        observation, _ = env.reset()
        done = False
        step_count = 0

        while not done:
            action = agent.act(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            agent.assign_reward(reward)
            done = terminated or truncated
            step_count += 1

        train_stats = agent.terminate_episode()
        wandb.log({
            "episode": episode,
            "episode_length": step_count,
            "total_return": sum(agent.current_episode_rewards) if agent.current_episode_rewards else step_count,
            **train_stats
        })
