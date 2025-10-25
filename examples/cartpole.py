import os
from tqdm import trange
import wandb
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

import re
import gymnasium as gym
from llamagym import Agent


class CartPoleAgent(Agent):
    def get_system_prompt(self) -> str:
        return """You are controlling a cart with a pole balanced on top. Your goal is to keep the pole upright by moving the cart left or right.

The pole starts upright. If it tilts too far (>12 degrees) or the cart moves too far off screen (>2.4 units), you lose.

Actions:
- Action: 0 = Push cart LEFT
- Action: 1 = Push cart RIGHT

Strategy: Move the cart in the direction the pole is falling to keep it balanced."""

    def format_observation(self, observation: gym.core.ObsType) -> str:
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        
        # Make the values more interpretable
        pos_desc = f"{'left' if cart_pos < -0.5 else 'right' if cart_pos > 0.5 else 'center'}"
        angle_desc = f"{'left' if pole_angle < -0.1 else 'right' if pole_angle > 0.1 else 'upright'}"
        
        return f"Cart at {pos_desc} (pos: {cart_pos:.2f}). Pole tilting {angle_desc} (angle: {pole_angle:.2f} rad, velocity: {pole_vel:.2f})."

    def extract_action(self, response: str) -> gym.core.ActType:
        # Look for "Action: 0" or "Action: 1"
        match = re.compile(r"Action:\s*(\d)").search(response)
        if match:
            action = int(match.group(1))
            if action in (0, 1):
                return action
        
        # Fallback: look for keywords
        response_lower = response.lower()
        if "left" in response_lower or "0" in response:
            return 0
        elif "right" in response_lower or "1" in response:
            return 1
        
        # Default to going right
        return 1


if __name__ == "__main__":
    hyperparams = {
        "model_name": "google/gemma-3-270m",
        "env": "CartPole-v1",
        "lora/r": 16,
        "lora/lora_alpha": 32,
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",
        "lora/target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "batch_size": 8,
        "seed": 42069,
        "episodes": 1000,  # CartPole episodes are shorter
        "generate/max_new_tokens": 32,
        "generate/do_sample": False,  # Greedy decoding for stability
        "generate/top_p": 0.6,
        "generate/top_k": 0,
        "generate/temperature": 0.9,
    }
    wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config=hyperparams)
    device = "cuda:0"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    lora_config = LoraConfig(
        **{
            key.split("/")[-1]: value
            for key, value in hyperparams.items()
            if key.startswith("lora/")
        }
    )
    
    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=hyperparams["model_name"],
        device_map={"": device},
        token=HF_TOKEN,
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"], token=HF_TOKEN)
    
    # Set pad token to eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set chat template for Gemma
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
    
    # Apply LoRA
    model = get_peft_model(base_model, lora_config).to(device)

    agent = CartPoleAgent(
        model,
        tokenizer,
        device,
        {key: value for key, value in hyperparams.items() if key.startswith("generate/")},
        {"batch_size": hyperparams["batch_size"]},
    )
    
    env = gym.make(hyperparams["env"])

    for episode in trange(hyperparams["episodes"]):
        observation, info = env.reset()
        done = False
        step_count = 0

        while not done:
            action = agent.act(observation)
            wandb.log({"action": action, "step": step_count})
            observation, reward, terminated, truncated, info = env.step(action)
            agent.assign_reward(reward)
            done = terminated or truncated
            step_count += 1

        episode_stats = {
            "episode": episode,
            "episode_length": step_count,
            "total_return": sum(agent.current_episode_rewards),
            "message_ct": len(agent.current_episode_messages),
        }
        train_stats = agent.terminate_episode()
        episode_stats.update(train_stats)
        wandb.log(episode_stats)
