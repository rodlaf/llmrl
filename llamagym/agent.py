from abc import ABC, abstractmethod
from typing import List, Dict

import gymnasium as gym
import torch
from trl import (
    GRPOTrainer,
    GRPOConfig,
)
from datasets import Dataset


class Agent(ABC):
    def __init__(
        self, model, tokenizer, device, generate_config_dict=None, training_config_dict=None
    ):
        if generate_config_dict is None:
            generate_config_dict = {
                "max_new_tokens": 32,
                "do_sample": True,
                "top_p": 0.6,
                "top_k": 0,
                "temperature": 0.9,
            }
        if training_config_dict is None:
            training_config_dict = {"batch_size": 16}

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generate_config_dict = generate_config_dict
        self.batch_size = training_config_dict.get("batch_size", 16)
        
        # Accumulate data from episodes for batch training
        self.batch_data = {"prompts": [], "completions": [], "rewards": []}
        self.grpo_trainer = None
        
        # Current episode state
        self.current_episode_messages = [{"role": "system", "content": self.get_system_prompt()}]
        self.current_episode_rewards = []

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def format_observation(self, observation: gym.core.ObsType) -> str:
        pass

    @abstractmethod
    def extract_action(self, response: str) -> gym.core.ActType:
        pass

    def llm(self, messages: List[Dict[str, str]]) -> str:
        self.model.eval()
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, 
                truncation=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                generate_ids = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **{k.split("/")[-1]: v for k, v in self.generate_config_dict.items()}
                )
                
            response = self.tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].split("[/INST]")[-1].strip()
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA error during generation: {e}. Attempting recovery...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                response = "Action: 0"  # Default fallback
            else:
                raise

        return response

    def act(self, observation):
        message = self.format_observation(observation)
        self.current_episode_messages.append({"role": "user", "content": message})

        response = self.llm(self.current_episode_messages)
        
        try:
            action = self.extract_action(response)
        except Exception:
            return None

        self.current_episode_messages.append({"role": "assistant", "content": response})
        return action

    def assign_reward(self, reward):
        self.current_episode_rewards.append(reward)

    def format_episode_for_training(self, messages, rewards):
        """Convert episode messages and rewards into query-response pairs."""
        queries, responses = [], []
        
        for i in range(2, len(messages), 2):
            prompt = self.tokenizer.apply_chat_template(
                messages[:i + 1], tokenize=False, add_generation_prompt=False
            )
            conversation_chunks = prompt.split("[/INST] ")
            query = "[/INST] ".join(conversation_chunks[:-1]) + "[/INST] "
            response = conversation_chunks[-1]

            queries.append(self.tokenizer(query, return_tensors="pt").input_ids[0])
            responses.append(self.tokenizer(response, return_tensors="pt").input_ids[0])

        # Handle sparse rewards (only final reward is non-zero)
        if all(r == 0 for r in rewards[:-1]):
            per_turn_reward = rewards[-1] / len(queries)
            rewards = [torch.tensor(per_turn_reward, dtype=torch.float16)] * len(queries)
        else:
            rewards = [torch.tensor(r, dtype=torch.float16) for r in rewards]

        return queries, responses, rewards

    def terminate_episode(self, train=True):
        if train:
            queries, responses, rewards = self.format_episode_for_training(
                self.current_episode_messages, self.current_episode_rewards
            )
            
            # Convert to text format and accumulate for batch training
            for query, response, reward in zip(queries, responses, rewards):
                self.batch_data["prompts"].append(
                    self.tokenizer.decode(query, skip_special_tokens=True)
                )
                self.batch_data["completions"].append(
                    self.tokenizer.decode(response, skip_special_tokens=True)
                )
                self.batch_data["rewards"].append(reward.item())

        # Reset episode state
        self.current_episode_messages = [{"role": "system", "content": self.get_system_prompt()}]
        self.current_episode_rewards = []

        # Train when batch is full
        if train and len(self.batch_data["prompts"]) >= self.batch_size:
            return self.train_batch()

        return {}

    def train_batch(self):
        """Train using GRPO with accumulated batch data."""
        # Extract batch
        prompts = self.batch_data["prompts"][:self.batch_size]
        completions = self.batch_data["completions"][:self.batch_size]
        rewards = self.batch_data["rewards"][:self.batch_size]
        
        # Store rewards for the reward function to access
        self._current_rewards = rewards
        
        # Create dataset
        dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
        
        # Reward function that returns our environment rewards
        def reward_function(prompts, completions, **kwargs):
            return self._current_rewards[:len(prompts)]
        
        # Initialize trainer on first batch
        if self.grpo_trainer is None:
            config = GRPOConfig(
                output_dir="./grpo_output",
                num_train_epochs=1,
                per_device_train_batch_size=self.batch_size,
                gradient_checkpointing=False,
                learning_rate=1e-5,
            )
            self.grpo_trainer = GRPOTrainer(
                model=self.model,
                reward_funcs=[reward_function],
                args=config,
                train_dataset=dataset,
                processing_class=self.tokenizer,
            )
        else:
            # Update for new batch
            self.grpo_trainer.train_dataset = dataset
            self.grpo_trainer.reward_funcs = [reward_function]
        
        # Train and collect stats
        try:
            self.grpo_trainer.train()
            stats = {
                "trained": True,
                "batch_rewards_mean": sum(rewards) / len(rewards),
                "batch_rewards_min": min(rewards),
                "batch_rewards_max": max(rewards),
            }
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Training error: {e}")
            stats = {"error": str(e)}
            torch.cuda.empty_cache()
        
        # Clear processed data, keep remainder
        if len(self.batch_data["prompts"]) > self.batch_size:
            self.batch_data = {
                "prompts": self.batch_data["prompts"][self.batch_size:],
                "completions": self.batch_data["completions"][self.batch_size:],
                "rewards": self.batch_data["rewards"][self.batch_size:]
            }
        else:
            self.batch_data = {"prompts": [], "completions": [], "rewards": []}
        
        return stats
