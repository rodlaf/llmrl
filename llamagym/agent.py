from abc import ABC, abstractmethod
import gymnasium as gym
import torch
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from datetime import datetime
import os
import wandb


class Agent(ABC):
    def __init__(self, model, tokenizer, device, generate_config, training_config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generate_config = generate_config
        self.batch_size = training_config["batch_size"]
        self.learning_rate = training_config.get("learning_rate", 1e-6)
        self.gamma = training_config.get("gamma", 0.99)  # Discount factor for returns
        
        self.batch_data = {"prompts": [], "completions": [], "rewards": []}
        self.grpo_trainer = None
        self.training_step = 0
        self.current_episode_messages = []
        self.current_episode_rewards = []
        
        # Track step count for detailed logging
        self.global_step = 0

    @abstractmethod
    def format_prompt(self, observation: gym.core.ObsType) -> str:
        pass

    @abstractmethod
    def extract_action(self, response: str) -> gym.core.ActType:
        pass

    def llm(self, prompt_text):
        self.model.eval()
        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Action:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, pad_token_id=self.tokenizer.pad_token_id, 
                                              eos_token_id=self.tokenizer.eos_token_id, **self.generate_config)
        
        new_tokens = generate_ids[0, input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Clean up immediately
        del inputs, generate_ids, new_tokens
        
        # Log response to wandb
        wandb.log({
            "step": self.global_step,
            "response": response,
            "response_length": len(response)
        })
        
        return response

    def act(self, observation):
        prompt_text = self.format_prompt(observation)
        print(f"Prompt: {prompt_text}")
        
        response = self.llm(prompt_text)
        action = self.extract_action(response)
        
        # Log interaction to wandb
        wandb.log({
            "step": self.global_step,
            "prompt": prompt_text,
            "action": action,
            "observation": str(observation)
        })
        
        self.current_episode_messages.append({"role": "user", "content": prompt_text})
        self.current_episode_messages.append({"role": "assistant", "content": response})
        
        self.global_step += 1
        return action

    def assign_reward(self, reward):
        self.current_episode_rewards.append(reward)

    def format_episode_for_training(self, messages, rewards):
        queries, responses = [], []
        for i in range(0, len(messages) - 1, 2):
            prompt = self.tokenizer.apply_chat_template([messages[i]], tokenize=False, add_generation_prompt=True)
            queries.append(prompt)
            responses.append(messages[i+1]["content"])
        
        # Calculate discounted returns for proper temporal credit assignment
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return queries, responses, returns

    def terminate_episode(self, train=True):
        if train:
            queries, responses, rewards = self.format_episode_for_training(
                self.current_episode_messages, self.current_episode_rewards)
            for query, response, reward in zip(queries, responses, rewards):
                self.batch_data["prompts"].append(query)
                self.batch_data["completions"].append(response)
                self.batch_data["rewards"].append(reward)

        self.current_episode_messages = []
        self.current_episode_rewards = []

        if train and len(self.batch_data["prompts"]) >= self.batch_size:
            return self.train_batch()
        return {}

    def train_batch(self):
        prompts = self.batch_data["prompts"][:self.batch_size]
        completions = self.batch_data["completions"][:self.batch_size]
        rewards = self.batch_data["rewards"][:self.batch_size]
        
        # Clear cache before training
        torch.cuda.empty_cache()
        
        self._current_rewards = [torch.tensor(r, dtype=torch.float32) for r in rewards]
        dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
        
        def reward_function(prompts, completions, **kwargs):
            return self._current_rewards[:len(prompts)]
        
        if self.grpo_trainer is None:
            config = GRPOConfig(
                num_train_epochs=1, 
                max_steps=1,
                per_device_train_batch_size=self.batch_size, 
                learning_rate=self.learning_rate, 
                logging_steps=1, 
                report_to=[],
                gradient_accumulation_steps=1,
            )
            self.grpo_trainer = GRPOTrainer(
                model=self.model, 
                reward_funcs=[reward_function],
                args=config, 
                train_dataset=dataset, 
                processing_class=self.tokenizer
            )
        else:
            self.grpo_trainer.train_dataset = dataset
            self.grpo_trainer.reward_funcs = [reward_function]
        
        self.grpo_trainer.train()
        self.training_step += 1
        
        # Aggressive cleanup
        del self._current_rewards
        torch.cuda.empty_cache()
        
        self.batch_data = {"prompts": [], "completions": [], "rewards": []}
        
        return {
            "trained": True, 
            "training_step": self.training_step, 
            "batch_rewards_mean": sum(rewards) / len(rewards), 
            "batch_size": len(rewards)
        }
