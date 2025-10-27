# Learning Through Interaction: Balancing a CartPole Through Online GRPO with Language Models

*October 26, 2025*

I trained a small language model to learn CartPole balancing in real-time through pure online reinforcement learning. No offline datasets, no expert demonstrations. The LLM learns to balance the pole by reasoning about physics in natural language and receiving feedback from the environment. This approach presents several novel technical challenges compared to traditional RL.

## Technical Challenges

Traditional RL agents receive clean numerical state vectors and output action probabilities. When using LLMs for RL, the problem becomes fundamentally different due to the text interface.

### Prompt Definition

Describing CartPole physics to a language model requires balancing informativeness with token efficiency. The state space consists of $(x, \dot{x}, \theta, \dot{\theta})$ where $x$ is cart position, $\theta$ is pole angle, and dots denote time derivatives. After experimentation, this format worked best:

```
Keep pole balanced on cart. Current state:
Cart position: 0.042, velocity: -0.123
Pole angle: 0.087 rad, angular velocity: -0.234
(Negative values = leftward, positive = rightward)

Your action: LEFT or RIGHT
```

The prompt design affects both token efficiency and model comprehension. Too much physics terminology confuses the model; too little context prevents understanding of the dynamics.

### Action Parsing

Extracting discrete actions from free-form text responses is non-trivial. LLMs generate varied outputs like "I think LEFT would be best here", "LEFT seems right", or "Going LEFT to counteract the rightward tilt". The solution searches for the last occurrence of "LEFT" or "RIGHT" in the response, handling most edge cases. However, this remains brittle compared to standard policy network outputs.

### Reward Assignment

The fundamental challenge lies in credit assignment. Traditional PPO operates on state-action pairs with value function bootstrapping:

$$V^\pi(s) = \mathbb{E}_{\pi}[\sum_{t=0}^{\infty} \gamma^t r_{t+1} | s_0 = s]$$

With LLMs, the mapping becomes $s \rightarrow \text{prompt} \rightarrow \text{response} \rightarrow a$. The reward must be assigned to the text generation process, not just the final action. Using GRPO (Group Relative Policy Optimization), the CartPole reward supervises the language modeling objective, creating coupling between environmental rewards and next-token prediction.

*[Graph 1: Learning curves showing episode length over time on single 4090]*

### Hardware Constraints

Training on a single RTX 4090 with 24GB VRAM requires careful memory management. The system must maintain model weights, gradients, GRPO trainer overhead, batch accumulation buffers, and tokenization structures simultaneously. Aggressive CUDA cache clearing after each forward pass prevents out-of-memory errors but impacts training speed.

*[Graph 2: Memory usage over time during training episodes]*

## Implementation 

The CartPoleAgent inherits from a custom Agent class that orchestrates language model inference and RL training:

```python
class CartPoleAgent(Agent):
    def format_prompt(self, observation):
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return f"""Keep pole balanced on cart. Current state:
Cart position: {cart_pos:.3f}, velocity: {cart_vel:.3f}
Pole angle: {pole_angle:.3f} rad, angular velocity: {pole_vel:.3f}
(Negative values = leftward, positive = rightward)

Your action: LEFT or RIGHT"""

    def extract_action(self, response: str):
        # Parse natural language response to discrete action
        response_upper = response.upper()
        left_pos = response_upper.rfind("LEFT")
        right_pos = response_upper.rfind("RIGHT")
        
        if left_pos != -1 and right_pos != -1:
            return 0 if left_pos > right_pos else 1
        elif left_pos != -1:
            return 0
        elif right_pos != -1:
            return 1
        else:
            return random.choice([0, 1])  # Fallback
```

The training loop operates on state→text→action mappings rather than direct state→action mappings. Traditional RL optimizes $\pi(a|s)$, while this approach optimizes the composition $\pi_{parse}(a|\text{response}) \circ \pi_{LLM}(\text{response}|\text{prompt}) \circ f(\text{prompt}|s)$ where $f$ is the prompt formatting function.

Temporal credit assignment becomes complex because environmental rewards must propagate back to text tokens generated several steps earlier. The discounted return calculation:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

applies to text completions rather than state-action pairs.

*[Graph 3: Action parsing success rate over different response formats]*

## Analysis

### Practical Implications

This approach offers interpretability advantages over traditional RL. The agent's reasoning process is human-readable, enabling easier debugging and analysis of failure modes. Pre-trained language models bring substantial prior knowledge about physics and spatial reasoning, potentially improving sample efficiency compared to training from scratch.

*[Graph 4: Comparison of sample efficiency vs traditional RL baseline]*

## Experimental Results

The 500M parameter Qwen2.5 model learns to balance the CartPole through online interaction. Training exhibits distinct phases: initial episodes show random responses and frequent parsing failures, middle episodes develop consistent action vocabulary with poor timing, and later episodes demonstrate basic control strategies.

The model occasionally develops linguistic patterns that don't correspond to optimal policies. For instance, it may learn to consistently output "LEFT" due to early positive experiences, even when "RIGHT" would be more appropriate for the current state.

*[Graph 5: Episode rewards and length over training time]*

## Future Directions

Multi-environment transfer represents an interesting research direction. The linguistic reasoning mechanisms might generalize better across different control tasks compared to standard policy networks. Scaling experiments with larger models could reveal whether the benefits plateau at smaller sizes or continue improving.

Interactive training through natural language feedback could accelerate learning. The system could potentially incorporate corrections like "Move opposite to the pole's lean direction" directly into the training process.

## Implementation

The code is available for replication:

```bash
git clone https://github.com/rodlaf/llmrl
cd llmrl
pip install -r requirements.txt
python examples/cartpole.py --model Qwen/Qwen2.5-0.5B-Instruct
```

Training requires a GPU with sufficient memory (tested on RTX 4090). Initial episodes may appear unproductive due to action parsing issues and random exploration.

## Conclusion

This experiment demonstrates the feasibility of using language models for online reinforcement learning in control tasks. While not competitive with specialized RL algorithms for CartPole, the approach offers unique advantages in interpretability and potential for human interaction. The technical challenges around prompt design, action parsing, and credit assignment reveal fundamental differences between traditional RL and language-model-based agents.

The code provides a foundation for further research into linguistic reasoning in RL contexts, though significant optimization would be required for practical applications.

Code available at: github.com/rodlaf/llmrl

## Citation

```bibtex
@misc{llmrl2024,
  title        = {LLMRL: Real-Time LLM Agents Learning Control Through Online Reinforcement Learning},
  author       = {Rodney Lafuente},
  year         = {2024},
  howpublished = {GitHub},
  url          = {https://github.com/rodlaf/llmrl}
}
```