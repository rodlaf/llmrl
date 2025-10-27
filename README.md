# Orlllam: Online Reinforcement Learning trained Large Language Models

*Fine-tune language models with online reinforcement learning*

## Overview

ORLLLaM demonstrates how small language models can learn control tasks through direct interaction with environments. Rather than relying on offline datasets or pre-computed demonstrations, LLMs learn by reasoning about states in natural language and receiving immediate feedback.

The primary contribution is showing that a 500M parameter language model can learn to balance a CartPole through online GRPO (Group Relative Policy Optimization), handling the unique challenges of text-based action spaces and credit assignment in language model RL.

## Key Features

- **Online Learning**: LLMs learn through direct environment interaction, not offline data
- **Natural Language Reasoning**: Agents process states and reason about actions in text
- **Small Models**: Demonstrates learning with lightweight models (500M parameters)
- **Text-Based Actions**: Handles the complexity of parsing discrete actions from free-form text
- **Memory Efficient**: Runs on single GPU (tested on RTX 4090)

## Installation

```bash
git clone https://github.com/rodlaf/llmrl
cd llmrl
pip install -r requirements.txt
```

## CartPole Example

The main demonstration shows an LLM learning to balance a CartPole:

```bash
python examples/cartpole.py --model Qwen/Qwen2.5-0.5B-Instruct
```

The agent receives state descriptions like:
```
Keep pole balanced on cart. Current state:
Cart position: 0.042, velocity: -0.123
Pole angle: 0.087 rad, angular velocity: -0.234
Your action: LEFT or RIGHT
```

And learns to generate appropriate responses through GRPO training on environmental rewards.

## Implementation

The core `Agent` class handles the complexity of:
- Converting environment states to natural language prompts
- Parsing discrete actions from LLM text responses  
- Assigning environmental rewards to text generation processes
- Managing memory efficiently during online training

```python
from orllam import Agent

class CartPoleAgent(Agent):
    def format_prompt(self, observation):
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return f"""Keep pole balanced on cart. Current state:
Cart position: {cart_pos:.3f}, velocity: {cart_vel:.3f}
Pole angle: {pole_angle:.3f} rad, angular velocity: {pole_vel:.3f}
Your action: LEFT or RIGHT"""

    def extract_action(self, response: str):
        response_upper = response.upper()
        left_pos = response_upper.rfind("LEFT")
        right_pos = response_upper.rfind("RIGHT") 
        return 0 if left_pos > right_pos else 1
```

## Technical Challenges

This approach presents several novel challenges compared to traditional RL:

**Prompt Engineering**: Balancing informativeness with token efficiency when describing physics
**Action Parsing**: Reliably extracting discrete actions from variable text responses
**Credit Assignment**: Mapping environmental rewards to text generation rather than state-action pairs
**Memory Management**: Training language models online with limited GPU memory

## Results

The 500M parameter model successfully learns CartPole balancing through online interaction. Training exhibits distinct phases from random responses to consistent action vocabulary to basic control strategies.

## Other Examples

- `examples/blackjack.py`: Blackjack strategy learning
- `examples/text-world.py`: Text-based adventure game interaction

## Citation

```bibtex
@misc{orllam2024,
  title        = {ORLLLaM: Online Reinforcement Learning Large Language Models},
  author       = {Rodney Lafuente},
  year         = {2024},
  howpublished = {GitHub},
  url          = {https://github.com/rodlaf/llmrl}
}
```

## License

MIT License


# LlamaGym
"Agents" originated in reinforcement learning, where they learn by interacting with an environment and receiving a reward signal. However, LLM-based agents today do not learn online (i.e. continuously in real time) via reinforcement.

OpenAI created [Gym](https://github.com/Farama-Foundation/Gymnasium) to standardize and simplify RL environments, but if you try dropping an LLM-based agent into a Gym environment for training, you'd find it's still quite a bit of code to handle LLM conversation context, episode batches, reward assignment, PPO setup, and more.

LlamaGym seeks to simplify fine-tuning LLM agents with RL. Right now, it's a single `Agent` abstract class that handles all the issues mentioned above, letting you quickly iterate and experiment with agent prompting & hyperparameters across any Gym environment.

## Usage
Fine-tuning an LLM-based agent to play in a Gym-style environment with RL has never been easier! Once you install LlamaGym...
```
pip install llamagym
```

First, implement 3 abstract methods on the Agent class:
```python
from llamagym import Agent

class BlackjackAgent(Agent):
    def get_system_prompt(self) -> str:
        return "You are an expert blackjack player."

    def format_observation(self, observation) -> str:
        return f"Your current total is {observation[0]}"

    def extract_action(self, response: str):
        return 0 if "stay" in response else 1
```

Then, define your base LLM (as you would for any fine-tuning job) and instantiate your agent:
```python
model = AutoModelForCausalLMWithValueHead.from_pretrained("Llama-2-7b").to(device)
tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b")
agent = BlackjackAgent(model, tokenizer, device)
```

Finally, write your RL loop as usual and simply call your agent to act, reward, and terminate:
```python
env = gym.make("Blackjack-v1")

for episode in trange(5000):
    observation, info = env.reset()
    done = False

    while not done:
        action = agent.act(observation) # act based on observation
        observation, reward, terminated, truncated, info = env.step(action)
        agent.assign_reward(reward) # provide reward to agent
        done = terminated or truncated

    train_stats = agent.terminate_episode() # trains if batch is full
```

Some reminders:
- above code snippets are mildly simplified above but a fully working example is available in [`examples/blackjack.py`](https://github.com/KhoomeiK/LlamaGym/blob/main/examples/blackjack.py)
- getting online RL to converge is notoriously difficult so you'll have to mess with hyperparameters to see improvement
  - your model may also benefit from a supervised fine-tuning stage on sampled trajectories before running RL (we may add this feature in the future)
- our implementation values simplicity so is not as compute efficient as e.g. [Lamorel](https://github.com/flowersteam/lamorel), but easier to start playing around with
- LlamaGym is a weekend project and still a WIP, but we love contributions!

## Relevant Work
- [Grounding Large Language Models with Online Reinforcement Learning](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)
  - [Lamorel: Language Models for Reinforcement Learning](https://github.com/flowersteam/lamorel)
- [True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning](https://github.com/WeihaoTan/TWOSOME)
