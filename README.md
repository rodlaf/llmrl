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
