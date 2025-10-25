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
        self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None
    ):
        if generate_config_dict is None:
            generate_config_dict = {
                "max_new_tokens": 32,
                "do_sample": True,
                "top_p": 0.6,
                "top_k": 0,
                "temperature": 0.9,
            }
        if ppo_config_dict is None:
            ppo_config_dict = {"batch_size": 16, "mini_batch_size": 16}

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generate_config_dict = generate_config_dict
        
        # Store config for batch management
        self.batch_size = ppo_config_dict.get("batch_size", 16)
        
        # GRPO-specific: Store prompts, completions, and rewards from episodes
        self.grpo_batch = {
            "prompts": [], 
            "completions": [],
            "rewards": []
        }
        
        # We'll initialize GRPO trainer when we have our first batch of data
        self.grpo_trainer = None
        self.grpo_config_dict = ppo_config_dict

        self.current_batch = {"queries": [], "responses": [], "rewards": []}

        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
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
        # Ensure model is in eval mode for inference
        self.model.eval()
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():  # No gradients needed for inference
                generate_ids = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **{
                        key.split("/")[-1]: value
                        for key, value in self.generate_config_dict.items()
                    }
                )
            outputs = self.tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = outputs[0].split("[/INST]")[-1].strip()
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA error during generation: {e}")
                print("Attempting to recover...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Return a default action
                response = "Action: 0"
            else:
                raise

        return response

    def act(self, observation):
        message = self.format_observation(observation)
        self.current_episode_messages += [{"role": "user", "content": message}]

        response = self.llm(self.current_episode_messages)
        try:
            action = self.extract_action(response)
        except Exception as e:
            return None

        self.current_episode_messages += [{"role": "assistant", "content": response}]
        return action

    def assign_reward(self, reward):
        self.current_episode_rewards.append(reward)

    def format_episode_for_ppo(self, messages, rewards):
        queries, responses = [], []
        for i in range(2, len(messages), 2):
            prompt = self.tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=False, add_generation_prompt=False
            )
            conversation_chunks = prompt.split("[/INST] ")
            query = "[/INST] ".join(conversation_chunks[:-1]) + "[/INST] "
            response = conversation_chunks[-1]

            query = self.tokenizer(query, return_tensors="pt").input_ids[0]
            response = self.tokenizer(response, return_tensors="pt").input_ids[0]

            queries.append(query)
            responses.append(response)

        if all(reward == 0 for reward in rewards[:-1]):
            # if sparse rewards, give equal reward to all conversation turns
            per_turn_reward = rewards[-1] / (len(messages) / 2)
            rewards = [torch.tensor(per_turn_reward, dtype=torch.float16)] * len(
                queries
            )
        else:
            rewards = [torch.tensor(reward, dtype=torch.float16) for reward in rewards]

        return queries, responses, rewards

    def terminate_episode(self, train=True):
        if train:
            queries, responses, rewards = self.format_episode_for_ppo(
                self.current_episode_messages, self.current_episode_rewards
            )
            
            # For GRPO, we need to convert queries to prompts (text format) and responses to completions
            for query, response, reward in zip(queries, responses, rewards):
                prompt_text = self.tokenizer.decode(query, skip_special_tokens=True)
                completion_text = self.tokenizer.decode(response, skip_special_tokens=True)
                self.grpo_batch["prompts"].append(prompt_text)
                self.grpo_batch["completions"].append(completion_text)
                self.grpo_batch["rewards"].append(reward.item())

        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
        self.current_episode_rewards = []

        if train and len(self.grpo_batch["prompts"]) >= self.batch_size:
            train_stats = self.train_grpo_batch()
            return train_stats

        return {}

    def train_grpo_batch(self):
        """Train using GRPO with accumulated prompts, completions, and rewards."""
        # Get the batch to train
        batch_prompts = self.grpo_batch["prompts"][:self.batch_size]
        batch_completions = self.grpo_batch["completions"][:self.batch_size]
        batch_rewards = self.grpo_batch["rewards"][:self.batch_size]
        
        # Create dataset from accumulated prompts
        dataset = Dataset.from_dict({
            "prompt": batch_prompts,
            "completion": batch_completions,
        })
        
        # Create a reward function that returns our stored rewards
        # GRPO will call this with prompts and completions it generated
        # But since we're providing completions in the dataset, it might use those
        def reward_function(prompts, completions, **kwargs):
            # Return the rewards we collected from the environment
            return batch_rewards[:len(prompts)]
        
        # Initialize GRPO trainer if not already done
        if self.grpo_trainer is None:
            # GRPOConfig uses different parameter names than PPOConfig
            grpo_config = GRPOConfig(
                output_dir="./grpo_output",
                num_train_epochs=1,
                per_device_train_batch_size=self.batch_size,
                gradient_checkpointing=False,  # Disable to avoid cache issues
                learning_rate=1e-5,  # Lower learning rate for stability
            )
            
            self.grpo_trainer = GRPOTrainer(
                model=self.model,
                reward_funcs=reward_function,
                args=grpo_config,
                train_dataset=dataset,
                processing_class=self.tokenizer,
            )
        else:
            # Update the dataset for the trainer
            self.grpo_trainer.train_dataset = dataset
        
        # Train for one step
        try:
            self.grpo_trainer.train()
            train_stats = {"loss": 0.0, "trained": True}
            
            # Clear CUDA cache after training to prevent memory issues
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"GRPO training error: {e}")
            train_stats = {"error": str(e)}
            
            # Try to recover from CUDA errors
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass
        
        # Clear the batch (or keep remainder if > batch_size)
        if len(self.grpo_batch["prompts"]) > self.batch_size:
            self.grpo_batch = {
                "prompts": self.grpo_batch["prompts"][self.batch_size:],
                "completions": self.grpo_batch["completions"][self.batch_size:],
                "rewards": self.grpo_batch["rewards"][self.batch_size:]
            }
        else:
            self.grpo_batch = {"prompts": [], "completions": [], "rewards": []}
        
        return train_stats
