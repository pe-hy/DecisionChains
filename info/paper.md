## Contents
- 1 Introduction
- 2 Related Work
- 3 Approach
  - 3.1 Preliminaries
  - 3.2 Strategy-Guided Exploration
  - 3.3 RL Details
- 4 Experiments
  - 4.1 Environments
  - 4.2 Baselines
  - 4.3 Empirical Comparison to Baselines
  - 4.4 Analysis
- 5 Conclusion and Limitations
- Appendix A Implementation Details
  - A.1 Prompts
  - A.2 Further SGE Details
  - A.3 Hyperparameters
  - A.4 Baseline Details
- Appendix B Additional Results
- Appendix C Environment Details
  - C.1 Coding
  - C.2 Android World
  - C.3 Language Rearrangement (LangR)
  - C.4 AppWorld

## Abstract

Abstract Reinforcement learning (RL) has demonstrated notable success in post-training large language models (LLMs) as agents for tasks such as computer use, tool calling, and coding. However, exploration remains a central challenge in RL for LLM agents, especially as they operate in language-action spaces with complex observations and sparse outcome rewards. In this work, we address exploration for LLM agents by leveraging the ability of LLMs to plan and reason in language about the environment to shift exploration from low-level actions to higher-level language strategies. We thus propose Strategy-Guided Exploration (SGE), which first generates a concise natural-language strategy that describes what to do to make progress toward the goal, and then generates environment actions conditioned on that strategy. By exploring in the space of strategies rather than the space of actions, SGE induces structured and diverse exploration that targets different environment outcomes. To increase strategy diversity during RL, SGE introduces mixed-temperature sampling, which explores diverse strategies in parallel, along with a strategy reflection process that grounds strategy generation on the outcomes of previous strategies in the environment. Across UI interaction, tool-calling, coding, and embodied agent environments, SGE consistently outperforms exploration-focused RL baselines, improving both learning efficiency and final performance. We show that SGE enables the agent to learn to solve tasks too difficult for the base model.

## 1 Introduction

Large language models (LLMs) are a promising foundation for building agents across a wide range of downstream tasks such as computer use , coding , tool usage , and robotics .
A key driver of progress in LLM agents is reinforcement learning (RL), which trains LLMs to autonomously act to solve tasks in external environments.
RL has shown the ability to teach LLMs to reason, self-correct, and interact with complex environments using simple, easy-to-specify rewards without human feedback .

Exploration is a central challenge in training LLM agents with RL.
This challenge is amplified by sparse outcome rewards, which reward the agent only when it achieves the desired goal.
In domains such as video game playing or navigation , RL can train superhuman policies starting from a base policy that performs poorly.
In contrast, RL for LLM agents operates in complex language action spaces such as code outputs or tool calls, and is initialized from a pretrained model that induces a concentrated policy over likely outputs.
As a result, RL for LLM agents predominantly samples from and refines behaviors already supported by the base model, limiting the agent’s ability to discover new successful trajectories.
Consistent with this, empirical evidence from prior work in non-agentic reasoning domains demonstrates that RL post-training struggles to learn new tasks and instead refines outputs to high-reward trajectories the model is already capable of generating .

To improve exploration in RL training for LLM agents, we introduce Strategy-Guided Exploration (SGE), which expands agent capabilities by exploring new tasks not solvable by the base model.
SGE leverages the ability of LLMs to plan and reason in natural language about the agent’s environment.
Instead of treating exploration as sampling environment actions, SGE first generates a language “strategy" and then produces the environment action conditioned on that strategy.
The strategy is a concise and specific natural language description of what to do to make progress toward the goal.
We show that with SGE, exploring in the space of strategies is easier than exploring in the space of actions, and that diverse strategies lead to better exploration in the environment, which increases the likelihood of success.
Since strategies are high-level language descriptions of intended outcomes, the LLM can use its reasoning abilities to generate many distinct strategies.
Conditioning on these strategies then guides the LLM to produce action sequences consistent with the strategy that explore different environment outcomes.

SGE introduces several techniques to generate diverse strategies during RL training to improve exploration.
SGE does more than simply prompt the LLM agent to first produce a strategy, and introduces techniques to maximize strategy diversity throughout training.
The first component of SGE is mixed-temperature sampling, which samples the strategy at a higher temperature than the rest of the token sequence.
Higher temperature for action sampling often produces different actions that achieve the same environment outcome, such as clicking the same UI element at different positions.
In contrast, increasing temperature in the higher-level strategy language space generates strategies that correspond to different outcomes.
The second component of SGE is strategy reflection, where the model reflects on successful and failed strategies from earlier in RL training.
This reflection helps ground strategy generation in the details of the environment dynamics and further diversify strategies from previous attempts.
See [Figure˜1](#S1.F1) for an overview of the SGE method.

Figure: Figure 1: Overview of Strategy-Guided Exploration (SGE). SGE improves reinforcement learning (RL) training on hard agentic tasks for which the base model fails, even after many attempts (left of figure). SGE addresses this by having the LLM policy output a language “strategy” and then conditioning the action generation on this strategy. The reasoning capabilities of the LLM enable it to output diverse strategies through the techniques of: (1) mixed-temperature sampling, where strategy tokens are sampled with higher temperature than remaining tokens, and (2) strategy reflection, where strategies are generated to be distinct from other strategies executed earlier in RL training. This enables SGE to explore to solve hard tasks that the base model is not capable of succeeding in (right of figure).
Refer to caption: 2603.02045v1/x1.png

Our experiments show that SGE solves challenging tasks during RL training that the base model cannot solve even after many attempts across the multi-step agentic environments of coding , UI control , tool-calling , and embodied agents .
SGE also outperforms exploration-focused RL baselines, achieving higher final performance and better learning efficiency.
For example, in the multi-turn coding environment, the starting LLM solves 69% of the problems *at least once* out of 2048 attempts (pass@2048=0.69).
The best performing RL baseline with a 64% final success rate (pass@1=0.64) is unable to surpass this ceiling in task pass rate as it attains no learning signal for the remaining difficult problems.
SGE, capable of exploring to solve harder problems, achieves a 73% success rate.
These training gains then translate to better generalization on unseen tasks.
We demonstrate the broad applicability of SGE by showing consistent improvements with SGE across four diverse environments involving different observation and action spaces.
We also ablate the SGE components of mixed-temperature sampling and strategy reflection, along with analyzing the impact of SGE across LLM size.
We qualitatively analyze how the LLM is able to generate and use diverse strategies to solve new tasks that are unsolvable by the base model.
Overall, our experiments show how SGE uses exploration to unlock scaling RL training to expand agentic capabilities on difficult tasks.

## 2 Related Work

Exploration is a long-standing central challenge of RL.
Simple forms of exploration based on action sampling struggle in sparse-reward or long-horizon problems that require complex exploration behaviors .
To address these shortcomings, some works model the uncertainty around the dynamics or state of the environment and use this as a learning signal to explore the environment .
Other methods introduce temporally extended exploration without explicit uncertainty estimation .
Hierarchical RL and options learning seek to better handle exploration over long-horizon tasks.
Like these works, SGE also focuses on complex exploration strategies but leverages the strengths of LLMs to do so.
Other works use language for hierarchical control, but do so for communication between planner and control policies , not to improve exploration for solving harder tasks as in SGE.

Exploration in RL is especially important for LLM-based policies as they typically operate in complex language action spaces, such as open-ended code generation, and are rewarded for achieving sparse outcomes, such as calling a sequence of tools that accomplishes a goal.
Prior work focusing on non-agentic reasoning tasks suggests that RL for LLM post-training primarily refines existing abilities of the base LLM instead of teaching the LLM to solve new problems that are unsolvable by the base model .
We show that the improved exploration from SGE helps exceed these limits in agentic tasks.
Works attempt to maintain high output entropy throughout training; however, these techniques are more designed to help training stability and less to promote exploration .
Other works directly train for policy diversity using a “pass@$k$” reward during training, where the model is rewarded if any attempt out of $k$ attempts succeeds .
These methods also try to prevent policy collapse, but still require at least one of the $k$ attempts to succeed under the base model to obtain a learning signal.
Unlike these works, our method incentivizes exploration in problems where the base model achieves minimal starting pass@$k$.

Other methods introduce exploration-specific techniques to LLM RL training.
uses the random network distillation framework to reward novel token sequences while directly rewards the policy entropy.
Our method also incentivizes exploration, but does so through language reasoning, and we demonstrate that it empirically outperforms these added exploration objectives.
Furthermore, these works focus on LLM reasoning for math or reasoning problems with a question-and-answer format, while we focus on agentic settings where an LLM agent takes a series of actions to achieve a goal.
adds a UCB exploration reward over the predicted final answer, but we focus on action spaces with an unbounded number of action possibilities.
uses a pretrained diversity classifier score as a reward, which we do not have in our multi-step agentic setting.
Other works use the pretrained knowledge of LLMs to provide policy feedback , which is orthogonal to how SGE uses the LLM to reason over strategies for better exploration in sparse reward settings.
Works also use LLMs to guide exploration in other non-LLM policies , whereas we focus on exploration in a single end-to-end LLM agent.

Other works on post-training LLMs for reasoning also introduce language abstractions or plans of reasoning steps or solutions.
uses distillation to first output a “reasoning abstraction", similar to a strategy in SGE.
However, SGE does not use a teacher model and introduces diversity-maximizing techniques in RL such as mixed-temperature sampling and strategy reflection.
also requires multiple expert models to generate different modes and focuses on distillation rather than online RL, like in SGE.
introduces a “hint" to guide policy responses, but this hint is generated from the ground truth answer.
Unlike these works, SGE operates in multi-step agentic settings where only the desired final outcome is specified, such as passing a set of tests, and the ground truth correct action response is unknown.

## 3 Approach

### 3.1 Preliminaries

We formalize the agentic tasks we focus on as Partially Observable Markov Decision Processes (POMDPs) with goal space $\mathcal{G}$, observation space $\mathcal{O}$, action space $\mathcal{A}$, environment transition function $\mathcal{T}$, and a reward model $R$. For brevity, we omit other elements of the MDP. In the settings we study, $\mathcal{G}$ is a natural language description of the goal, $\mathcal{O}$ is a textual or visual input, and $\mathcal{Y}$ is the space of any textual outputs by the LLM.
The environments we consider involve actions like tool-calling and coding, thus the action space $\mathcal{A}\subset\mathcal{Y}$ is expressed as natural language.
The reward model $R$ is a sparse reward assigned at the end of the episode for successfully completing the task.

We consider an LLM policy $\pi:\mathcal{G}\times\mathcal{O}\rightarrow\mathcal{Y}\times\mathcal{A}$ with the goal of maximizing the POMDP cumulative reward. At the episode start, the LLM receives a goal $g\sim\mathcal{G}$ and starting observation $o_{1}\in\mathcal{O}$. First, the LLM policy samples an intermediate reasoning trace (chain of thought) $y_{1}\sim\pi(\cdot\mid g,o_{1})$ and then produces the action $a_{1}\sim\pi(\cdot\mid g,o_{1},y_{1})$ which is executed in the environment and returns the next observation $o_{2}=\mathcal{T}(o_{1},a_{1})$.
This processes repeats to sample a trajectory $o_{1},y_{1},a_{1},o_{2},y_{2},a_{2},\dots,a_{T},o_{T},r_{T}$ where $r_{T}$ is the outcome reward.
The objective is to train the LLM policy with RL to maximize the expectation of rewards over $\mathcal{G}$.

### 3.2 Strategy-Guided Exploration

A key challenge in RL is learning from sparse reward functions that only score the binary success of achieving the goal.
LLM agents operating in complex environments and language action spaces are unlikely to stumble onto the goal by chance, thus the starting LLM must already have the capability to intelligently search to find the goal.
Prior work shows that this reliance on the base model’s sampling prevents discovering truly new solutions .

We thus introduce Strategy-Guided Exploration (SGE) to incentivize the LLM policy to explore and discover solutions to tasks unsolvable by the starting policy.
The key idea of SGE is for the policy to produce high-level language strategies before actions, and then enforce diverse strategy sampling to drive exploration.
SGE does not require ground truth solutions, privileged information or access to a stronger LLM, and instead modifies the LLM sampling process during RL training.
Specifically, SGE makes three modifications over standard RL training: (1) strategy prompting, (2) mixed-temperature sampling, and (3) strategy reflection.

Strategy Prompting:
For goal $g\in\mathcal{G}$ and observation $o_{t}$ at episode step $t$, instead of sampling an intermediate reasoning trace $y_{t}\sim\pi(\cdot\mid g,o_{t})$, we first sample a strategy $s_{t}\sim S_{\pi}(\cdot|g,o_{t})$ from “strategy sampling distribution" $S_{\pi}$.
For clarity, we subsume the goal and observation $(g,o_{t})$ into just the observation $o_{t}$.
The intermediate reasoning trace and then action is generated with this strategy, meaning the full output distribution is $\pi(a_{t}|y_{t},s_{t},o_{t})\pi(y_{t}|s_{t},o_{t})S_{\pi}(s_{t}|o_{t})$.
The strategy is a concise language description that maps to a specific and non-overlapping distribution of actions.
As we will empirically demonstrate, $S_{\pi}$ induces actions that explore the environment more efficiently than general intermediate reasoning or direct action sampling.
The strategy is different from the other intermediate chain-of-thought text or action tokens in that it is sampled from the different distribution $S_{\pi}$, which uses a specific prompt and two additional techniques for generating diverse strategies described later in this section.

Specifically, for a given observation, SGE modifies the prompt to include text like: *First, give a strategy of what action to take after ‘Strategy:’ then generate the action after ‘Action:’*.
Simply modifying the prompt to produce a strategy before the action is not particularly novel, with prior works also showing the benefits of first producing a strategy-like output before the response .
The key difference in SGE is how the strategy is leveraged for exploration in RL with the next two techniques.

Mixed-Temperature Sampling:
This generates strategies at a higher token sampling temperature than typical for the model inference.
This higher temperature only affects the tokens sampled from the strategy distribution $S_{\pi}$ and not the remaining LLM outputs from $\pi$.
This means that the model response is decoded with a mixed-temperature, where the initial strategy is decoded with a high temperature and the remaining tokens are decoded with a comparatively lower temperature.
The insight is that diverse strategies will more effectively explore the environment than directly sampling more diverse actions.
Sampling actions with a high temperature can result in actions that are only superficially distinct.
For example, in UI control, high-temperature action sampling can result in the agent tapping at different coordinates on the same button.
This added noise could hinder RL if, for example, the UI agent’s noisy tap locations sometimes click off the intended button.
We empirically find it is easier for the model to sample different strategies, and then condition the lower-temperature action generation on these strategies.
To explore different strategies, SGE produces $K$ parallel strategies per task, which has no added generation overhead over the group-based RL algorithms typically used in LLM RL training, such as GRPO or RLOO , which require $K$ parallel responses per task for advantage estimation.

Strategy Reflection: During RL training, SGE also improves the diversity of strategy generation based on the environment feedback.
With some probability, SGE does “Negative Strategy Reflection”.
If a strategy fails (meaning the outcome reward $r_{T}=0$), then we condition a subsequent rollout in the same task on the failed strategy with a negative reflection prompt instructing the policy to critique the failed strategy and to generate a new strategy that addresses the flaws of the failed strategy.
If the environment also provides episode-level textual feedback in addition to the scalar reward $r_{T}$, we also include this with the strategy.
For example, in the case of coding, the output of a failed program provides textual output of failed tests or runtime errors.
Likewise, SGE also incorporates “Positive Strategy Reflection” where, with some probability, strategy generation is conditioned on a strategy from the same task that ended in success.
The prompt instructs the agent to produce a new strategy inspired by the successful example in the same task.
We find that the negative reflection enables the policy to solve new episodes beyond parallel strategy sampling.
Positive reflection results in better learning efficiency as the agent is able to better leverage successful examples and maintain higher output entropy with multiple successful strategies solving the same task.
Since the reflection process happens during RL, the strategies are from old versions of the policy.
This further boosts strategy diversity over the strategies purely generated from a single policy instance.

We detail all the prompts, including the strategy and reflection prompts in [Section˜A.1](#A1.SS1).
All prompts are short at mostly two sentences and are consistent between environments with slightly different language in how actions are referred to depending on the environment and LLM being trained.
Environments are multi-step, so we generate a per-action strategy at every step.
In strategy reflection, we condition the reflection process on the entire sequence of strategies from the previous episode.
We only use strategy reflection at training time, and evaluate the SGE trained policy in a standard setting.
Our evaluation domains have on the order of 10’s of steps per episode, which keeps this extra context manageable.
Future work extending SGE to environments with more steps could summarize long sequences of per-step strategies into a more compact meta-strategy to save the cost of the extra context.

### 3.3 RL Details

We use SGE with GRPO .
However, since SGE only alters the sampling distribution via mixed-temperature sampling and policy conditioning via strategy reflection, it is compatible with any online RL algorithm.
Between inference and training, our implementation uses a consistent per-token temperature, which varies between the strategy and remaining tokens, to ensure consistency in the policy distribution between the two stages.
We also use the GRPO modifications proposed by DAPO , except without the clip higher probability ratio, which we found was broadly harmful across all our experimental domains and baselines.
We detail all method details, prompts, and hyperparameters in [Appendix˜A](#A1).

Figure: (a) Coding
Refer to caption: 2603.02045v1/x2.png

## 4 Experiments

### 4.1 Environments

We show results across four diverse agentic domains to demonstrate the broad applicability of SGE.

- •
AndroidWorld is a phone UI control environment where the agent operates from RGB pixel inputs and outputs low-level UI interactions, including tapping pixel coordinates, swiping, or typing text.
We use 26 of the tasks for training and evaluate on 30 test tasks consisting of non-visual question answering tasks in the overall AndroidWorld task set.
Both the train and test sets span 15 apps, with the test set including tasks on 4 apps unseen in the training set.
- •
Language Rearrangement (LangR) : An embodied household agent is instructed to rearrange objects in a home environment to complete a language instruction, such as “Find and put all the apples in the fridge”. We use a textual environment observation, which is a partially observable textual representation of the robot’s vicinity, including the object the robot is currently holding, and which objects and receptacles are in front of the robot. The action space includes high-level skills such as picking objects and navigating to receptacles. We use the standard train split and evaluate generalization to unseen houses.
- •
Coding : Given a coding problem description, the agent must output Python code to pass a set of unit tests.
We use only the “hard” category of problems from the dataset, giving 606 tasks from the train split and 228 tasks from the test split.
We make this environment multi-turn by providing the output of the executed program as an observation for the policy to produce updated Python code in another attempt.
- •
AppWorld : A multi-step tool-calling benchmark in a simulated app ecosystem across 9 applications and 457 APIs.
We only train and test on the “Easy” categorization of problems, which are still challenging for the LLMs we finetune.
This filtering results in 36 training problems, and we evaluate generalization to unseen tasks with the 57 tasks from the “Test Normal” split.

We train separate models for each environment. See [Appendix˜C](#A3) for full details on all of the environments.

### 4.2 Baselines

We compare SGE against the following approaches for LLM RL training, with a focus on baselines that are intended to improve exploration.

- •
GRPO: Regular GRPO with the standard policy gradient objective.
- •
Entropy Advantage (EntropyAdv) : Modifies GRPO to incentivize exploration by adding the policy entropy to the advantage for every token.
This approach was demonstrated to be more effective for LLM RL exploration than standard entropy regularization that adds an entropy loss term to the RL objective .
- •
Random Network Distillation (RND) : Adds an exploration reward for producing novel actions using random network distillation (RND) . RND trains a network to predict the output of a randomly initialized frozen target network based on the final LLM activation for each action. The prediction error is provided as an exploration reward when the agent fails to achieve the goal, encouraging the model to explore novel output sequences.
- •
RL with Abstraction Discovery (RLAD) : This work first outputs an abstraction for the problem, akin to the strategy in SGE, and then conditions the response on the abstraction. RLAD trains the abstraction and solution generator by augmenting training with a KL divergence penalty and sometimes dropping out the strategy. We use the same LLM weights for both roles of abstraction and solution generator. RLAD serves as a baseline also addressing explicit strategy generation.
- •
Base model pass@k: This zero-shot evaluates the base model with $k$ independent attempts per task and is the percentage of the time that any of the attempts are successful as judged by the ground truth task verifier. This shows how methods improve on the existing capabilities of the base model.

We find prompting the policy to produce a strategy before the action to be marginally useful for even non-SGE baselines; therefore, all baselines use this improved prompt for a fair comparison (see [Appendix˜B](#A2) for more details).
Thus, the difference between SGE and the standard GRPO baseline is that SGE uses the mixed-temperature sampling and strategy reflection.
All methods, SGE included, are trained with the GRPO.
For LangR and Coding, we finetune the Qwen3-4B-Instruct model .
For AppWorld, we finetune the Qwen3-8B reasoning model, as we found the larger size was necessary for stable GRPO training.
For AndroidWorld, which has visual observations, we finetune the Qwen2.5-VL-3B model .
We only run the RND and RLAD baselines on Coding since they underperform the GRPO and EntropyAdv baselines.
All hyperparameters and full implementation details are provided in [Appendix˜A](#A1).

Note that since we operate in an agentic setup where an agent needs to take actions to achieve a goal, we cannot apply some of the methods from prior works that are specific to single-step reasoning question answer domains.
We cannot apply the UCB exploration method from since there is an unbounded number of action trajectories that can lead to the goal.
The agentic environments we evaluate in also do not contain reference solutions, so we cannot compare to the method from , which evaluates in the math domain and assumes access to the problem answers. assumes access to a pretrained diversity classifier, which is difficult to apply to strategy and action sequences.

Figure: (a) Coding
Refer to caption: 2603.02045v1/x7.png

### 4.3 Empirical Comparison to Baselines

RL training results:
We compare SGE RL training to baselines in [Figure˜2](#S3.F2), which demonstrates SGE achieves higher RL training performance.
Specifically, we plot the average pass@1 rate, which is the percentage of the time the agent succeeds in solving the problem on the train set, along with the standard error across 3 random training seeds in RL training.
The overlaid horizontal lines indicate the pass@$k$ of the base model for $k$ values up to where the pass@$k$ does not change when doubling $k$.
This means the base LLM cannot solve any more tasks, even with twice as many attempts.
The plots demonstrate that SGE achieves higher final training performance than baselines across all environments, with on average SGE achieving 27% higher relative final success than the next best baseline for that environment.
Notably, baselines are unable to surpass the maximum pass@$k$ performance of the base model, confirming the findings from works such as for agentic settings.
However, in Coding and LangR, SGE surpasses the maximum pass@$k$ by on average 11% relative performance, demonstrating that the exploration from SGE enables the model to learn new behaviors not demonstrated by the base model.

SGE also outperforms the EntropyAdv and RND baselines, which also focus on better exploration in LLM RL.
We find that while EntropyAdv and RND increase policy token output entropy in RL training, unlike SGE, they cannot explore to exceed the maximum pass@$k$.
These methods were developed in the context of question answering in math reasoning problems, while we focus on agentic exploration, where diversity over outcomes is more important than diversity over token outputs.
Different output tokens may correspond to actions that produce similar effects, for example, the agent may make only minor variations to the syntax of the code in the Coding environment or tap the same button at different positions in AndroidWorld.
SGE also outperforms RLAD, another method that leverages strategies.
RLAD is hampered by slow training, potentially due to the KL divergence penalty required in the method.
RLAD also only showed results for non-agentic question answering problems for the smaller Qwen3-1.7B model.
SGE better leverages strategy generation for exploration in agentic RL.

Next, in [Figure˜3](#S4.F3) we plot the pass@$k$ curve of the trained policy across SGE and baselines.
Regular GRPO training “flattens” the pass@$k$ curve, raising the pass@1 value with RL to closer match the pass@k value.
On the other hand, SGE benefits from increased test time scaling by increasing in performance with increased $k$ to consistently surpass the base model.
This indicates that SGE enables the policy to solve new tasks that are not solvable by the base model, even for very large $k$, where the pass@$k$ has plateaued for the base LLM.
This also occurs in AndroidWorld and AppWorld, where the RL pass@1 does not surpass the base model max pass@$k$.

**Table 1: Test evaluation on unseen tasks of SGE versus baselines after RL training. Numbers are average and standard error pass@1 across the 3 random seeds on unseen tasks not included in RL training. All coding checkpoints are evaluated after 1k updates. Zero-Shot refers to evaluating the base-LLM directly on the test set without any RL training. RL improves generalization, and SGE further boosts it.**
|  | Zero-Shot | GRPO | SGE |
| --- | --- | --- | --- |
| Coding | 13.5 | 22.0 $\pm$ 0.3 | 29.2 $\pm$ 0.7 |
| AndroidWorld | 16.7 | 21.9 $\pm$ 0.3 | 36.7 $\pm$ 1.3 |
| LangR | 42.6 | 46.0 $\pm$ 0.4 | 60.8 $\pm$ 0.6 |
| AppWorld | 47.8 | 49.3 $\pm$ 5.0 | 66.6 $\pm$ 2.9 |

Generalization to unseen tasks:
Next, in [Table˜1](#S4.T1), we evaluate the generalization of the RL-trained policies to unseen task instances using the test splits earlier described in [Section˜4.1](#S4.SS1).
The results in [Table˜1](#S4.T1) show that SGE trains policies that generalize to new problems better than baselines across the four considered environments.
This shows that SGE not only explores the train environments better but also learns generalizable behaviors that transfer to unseen tasks, indicating that the agent is able to utilize its ability to solve harder problems during training time on the unseen test tasks.

Figure: (a) Mixed-Temp Ablation
Refer to caption: 2603.02045v1/x11.png

### 4.4 Analysis

This section further analyzes SGE in a setup that focuses on difficult tasks where the base model struggles.
Specifically, we filter the hard category of the coding environment from [Section˜4.3](#S4.SS3) to only the problems where pass@256 of the Qwen3-4B model is $0\%$, which leaves $144$ problems.
These difficult problems test the limit of the model’s ability to explore and find creative solutions.
For this section, we finetune the Qwen3-8B model as it has more efficient RL training on these difficult problems for running analyses.

SGE component ablation: First, we analyze the contribution of the SGE components in [Figure˜4](#S4.F4).
In [Figure˜4(a)](#S4.F4.sf1), we remove the mixed-temperature sampling from SGE and instead sample all tokens with the default action temperature (“No Mixed - Default”) or the high temperature used to sample the strategy (“No Mixed - High”).
[Figure˜4(a)](#S4.F4.sf1) shows that either using the default or higher temperature results in worse performance than the mixed-temperature sampling.
[Figure˜4(b)](#S4.F4.sf2) shows that the failed strategies are more important in the reflection process than the successful strategies.
Both types of reflection help the overall SGE performance.
Overall, all components of SGE are important for the best performance.

Effect of mixed-temperature sampling on the base model: In [Figure˜5(a)](#S4.F5.sf1), we further analyze the effect of mixed-temperature sampling by comparing different temperature values for the strategy and the remaining tokens when sampling from the base model. [Figure˜5(a)](#S4.F5.sf1) shows that a high strategy temperature and a relatively lower temperature on the remaining tokens produce the highest pass@16, and thus the strongest exploration results. For RL training, we primarily care about the exploration capabilities of the model; thus, we use this mixed-temperature setting in SGE.

Effect of model scaling: In [Figure˜4(c)](#S4.F4.sf3), we analyze the effect of scaling the base LLM parameter count on RL training with or without SGE by training with 600M, 4B, and 8B Qwen3 model sizes.
Expectedly, training greatly improves with larger models, with the positive effects of SGE being most prominent at larger model scales.
SGE improves training at the 4B and 8B model scales, however, the smallest 600M model achieves close to $0\%$ success and SGE only barely helps.
This indicates that SGE is still limited by the level of reasoning and planning capability in the LLM, and if the base LLM is not capable of generating or leveraging diverse strategies, SGE, like the GRPO baseline, struggles to explore.

Figure: (a) Pass@16 vs. Temperature
Refer to caption: 2603.02045v1/x14.png

SGE qualitative exploration capabilities: In [Figure˜5(b)](#S4.F5.sf2), we visualize a qualitative example showing the exploration capabilities of SGE over standard GRPO training in the AndroidWorld *MarkorCreateNote* task.
In the visualized observation that occurs in the middle of the episode, the agent has already opened the new file dialogue and entered the filename.
Now, the agent must change the file extension from *.md* to *.txt*.
The QwenVL-2.5-3B model struggles with this step as it frequently tries to change the extension by typing *.txt* into the filename, while only selecting the other file extension in the dropdown actually changes the extension.
The left screenshot of [Figure˜5(b)](#S4.F5.sf2) shows that the tap locations of the regular policy sampling result in tapping slightly different locations on the filename extension.
However, the right screenshot of [Figure˜5(b)](#S4.F5.sf2) shows that SGE policy sampling taps several locations around the file creation dialogue box, and correctly interacts with the file extension dropdown menu.
Each of these different taps is driven by a different strategy.
While many of the taps are not correct, the correct tap is covered, thus increasing the pass rate for this task as previously empirically demonstrated in [Figure˜5(a)](#S4.F5.sf1).

See [Appendix˜B](#A2) for additional experimental results on the impact of the strategy prompt, qualitative examples of the strategy generation, and further analyzing exploration in SGE.

## 5 Conclusion and Limitations

This work introduces Strategy-Guided Exploration (SGE), a method for exploration in RL training for LLM agents.
SGE uses the textual reasoning abilities of LLMs to produce diverse high-level action strategies and then conditions the action generation on these textual strategies.
These strategies provide an effective space for agents to explore diverse outcomes in the environment.
SGE uses mixed-temperature sampling and strategy reflection to improve the diversity of generated strategies and thus achieve better exploration to solve challenging tasks.
We empirically demonstrate that SGE outperforms prior RL approaches across a variety of agentic domains in coding, UI control, tool-calling, and embodied AI.

A limitation of SGE is that it requires the starting LLM to have sufficient reasoning and planning capacity to generate and leverage the strategies.
We design SGE for agentic scenarios where the agent interacts with an external environment, and leave extending SGE for other non-agentic RL scenarios, like math reasoning training, for future work.
SGE also introduces the additional cost of having to produce a strategy before every step. This limits real-world deployment where response latency is important.
Future work can investigate how to dynamically predict new strategies only when necessary.

## Appendix A Implementation Details

### A.1 Prompts

SGE requires setting three prompts: (1) the *strategy prompt* which guides the LLM to first generate a strategy before the remaining tokens, (2) the *positive reflection prompt* which conditions the strategy generation on previously executed successful strategies, and (3) the *negative reflection prompt* which conditions the strategy generation on previously executed failed strategies.
The prompts we use for the different environments are largely the same.
Differences are that the action is referred to differently in the environments as “code”, “tool call”, or “action”.
Additionally, the AndroidWorld prompt required formatting the strategy as XML tags to follow the Qwen UI prompt format , whereas the other environments use markdown formatting.
In the reflection prompts, the strategy to reflect on is inserted in {strats}.

Coding environment:

- •
Strategy prompt: First give a strategy of how to solve the question after ‘‘### Strategy’’. Then write the code to solve the question based on the strategy and question in ‘‘### Code’’.
- •
Positive reflection prompt: First give a strategy of how to solve the question after ‘‘### Strategy’’ inspired by this successful strategy.{strats} Then write the code to solve the question based on the strategy and question in ‘‘### Code’’.
- •
Negative reflection prompt: First, after ‘‘### Strategy’’ critique the failed strategy and how it can be fixed. Be precise. Then address this critique by writing a better strategy. Make sure the strategy is detailed, and the code is easy to implement from the strategy. {strats} Then write the code to solve the question based on the strategy and question in ‘‘### Code’’.

AndroidWorld environment:

- •
Strategy prompt: For each function call, return a strategy in the <strategy></strategy> XML tags and a json object with function name and arguments within <tool_call></tool_call> XML tags.
- •
Positive reflection prompt: Here is a previous strategy that was successful at each step: {strats}
- •
Negative reflection prompt: Here are the previous sequence of strategies that failed: {strats} Your new strategy must be different from this failed strategy sequence and try something new.

LangR environment:

- •
Strategy prompt: First give a strategy of how to solve the question after ‘‘### Strategy’’. Then output the tool call action after ‘‘### Action’’.
- •
Positive reflection prompt: Here is a previous strategy that was successful at each step: {strats} First give a strategy of how to solve the question after ‘‘### Strategy’’ inspired by this successful strategy. Then generate the tool call in ‘‘### Action’’.
- •
Negative reflection prompt: Here is a previous strategy that failed: {strats} First, after ‘‘### Strategy’’ critique the failed strategy and how it can be fixed. Be precise. Then address this critique by writing a better strategy. Make sure the strategy is detailed and the code is easy to implement from the strategy. Then generate the tool call in ‘‘### Action’’.

AppWorld environment:

- •
Strategy prompt: Solve this step by step. First think about what your next step should be. Then write the code to execute that step.
- •
Positive reflection prompt: Here is a previous approach that was successful at each step:\n{strats}\n Inspired by this successful approach, first think about what your next step should be. Then write the code to execute that step.
- •
Negative reflection prompt: Here is a previous approach that failed:\n{strats}\nFirst, think and critique the failed approaches and propose how it can be fixed. Be precise. Then address this critique by writing a better approach. Make sure the approach is detailed and the code is easy to implement from the approach. Once you are done thinking, write the code to execute your next step.

In AppWorld, we use the reasoning model and take the entire content in <think>...</think> as the strategy.
We use the non-reasoning mode for LangR and coding, meaning the model does not first output the <think>...</think> block.
Instead, we extract the strategy from the “Strategy" markdown block part of the prompt described in [Section˜A.1](#A1.SS1).
Note that even without the explicit “think" block, the model still produces textual reasoning and a strategy outside of the action.
While we demonstrate SGE works both with and without reasoning mode enabled, we use the non-reasoning mode on LangR and Coding as it is more efficient to train since it requires fewer tokens per action, yet it is still sufficient to reason over strategy generation.
Qwen2.5-VL does not have any reasoning model, so we extract the strategy from the prompted <strategy>...</strategy> XML tag.
Note that all methods and baselines use the strategy prompt for all environments, while SGE with strategy reflection uses the positive and negative reflection prompts.

**Table 2: RL hyperparameters across domains. Unless otherwise specified, these values are shared between all approaches. SGE prefix means the value only applies to SGE, and not the baselines. *# Output Tokens / Step* refers to the total token budget per LLM output for each step in the environment. This multiplies the number of steps in an episode by the total number of tokens produced in an episode. The *Global Update Batch Size* is in terms of the number of steps, not the number of trajectories, which consist of multiple steps. *Group Size* is the number of trajectories used to estimate the advantage in GRPO. *Trajectories per update* is the number of samples collected in the environment for each update. *Token Temperature* is the temperature used to generate the non-strategy tokens.**
|  | Code | AndroidWorld | LangR | AppWorld |
| --- | --- | --- | --- | --- |
| Learning Rate | 1e-6 | 1e-6 | 1e-6 | 1e-5 |
| Group Size | 16 | 6 | 16 | 8 |
| Global Update Batch Size | 128 | 64 | 128 | 128 |
| # Output Tokens / Step | 2048 | 564 | 520 | 768 |
| Clip Param | 0.2 | 0.2 | 0.2 | 0.2 |
| Trajectories Collected Per Update | 1024 | 192 | 512 | 128 |
| Per-Update Epochs | 2 | 2 | 1 | 2 |
| Token Temperature | 0.7 | 0.7 | 0.6 | 0.7 |
| SGE Strategy Temperature | 1.2 | 1.2 | 1.0 | 1.2 |
| SGE Strategy Buffer Size | 32 | 32 | 32 | 32 |
| SGE Failure Strategy Reflection Prob | 0.25 | 0.25 | 0.25 | 0.25 |
| SGE Success Strategy Reflection Prob | 0.1 | 0.1 | 0.1 | 0.5 |

Figure: Figure 6: We report the impact of SGE on the zero-shot performance of the policy in the Coding environment and compare to the base model and SGE RL trained model. We evaluate in the same setup as [Figure˜3](#S4.F3).
Refer to caption: 2603.02045v1/x16.png

### A.2 Further SGE Details

See [Algorithm˜1](#alg1) for SGE pseudocode. Lines 5-10 show how the policy inference during data collection is augmented with the mixed-temperature sampling and strategy reflection. Otherwise, SGE follows typical GRPO training. The buffers are implemented as a first-in-first-out buffer to keep the most on-policy data.

### A.3 Hyperparameters

[Table˜2](#A1.T2) breaks down the hyperparameters for RL training for each environment.
As [Table˜2](#A1.T2) demonstrates, the values are largely the same between the domains.
Shared RL hyperparameters were first tuned by running regular GRPO in the environment.
Some of the settings relating to the batch size changed in each environment due to the different required context lengths.
The number of output tokens for AndroidWorld and LangR was selected by taking the maximum action length for that environment and adding 512 additional tokens for non-action intermediate outputs for chain-of-thought or strategy generation.
We use a constant learning rate throughout training, and do not employ any learning rate scheduler.
The RL parameters in the top section of [Table˜2](#A1.T2) are shared between all RL methods.
Each RL training job across all environments was run over 16 H100s. The number of updates used to train each method in each environment is displayed in the x-axis of [Figure˜2](#S3.F2).
For all the experiments, we train all LLM parameters, including the LLM vocabulary embedding and output layers.
For the Qwen2.5-VL experiments, we freeze the visual encoder module and only train the LLM components.

### A.4 Baseline Details

Entropy Advantage: This method shapes the per-token advantage term with the token entropy. Specifically, for advantage estimate $A_{t}$ at output token $o_{t}$ with output distribution entropy $\mathcal{H}_{t}$, the advantage term is modified as:

$$ $\displaystyle A_{t}^{\prime}=A_{t}+\min\left(\alpha\cdot\mathcal{H}_{t},\frac{|A_{t}|}{\kappa}\right)$ $$

Where $\alpha$ is the scaling coefficient and $\kappa$ controls the clipping threshold. As in , we set these to $\kappa=2$ and $\alpha=0.4$. Note that the entropy $\mathcal{H}_{t}$ is detached from the computational graph. This shaping ensures that the entropy does not dominate the advantage term or reverse the sign of the advantage.

RL with Abstraction Discovery (RLAD): Unlike the other baselines and SGE, this approach uses a KL divergence term to the starting policy. We set the KL divergence loss coefficient to be $0.001$. We skip conditioning the solution generation on the strategy $25\%$ of the time in RL training. Note that, unlike , we implement RLAD in a single model where the abstraction and solution generation are performed by a single model.

Random Network Distillation (RND): RND adapted for LLM post-training is referred to as “i-MENTOR” in . To adapt this method to a multi-step decision-making formulation, we use the final token activation for each action in the output sequence for the RND prediction task.
Specifically, let $y_{t,1},\dots,y_{t,L}$ be the sequence of output final activations, meaning the final hidden state before the LLM token logit projection head, for step $t$ in the environment, and which are then decoded into action $a_{t}$.
This hidden state is passed through a randomly initialized MLP, $\overline{f}$, to form a target $\overline{z}=\overline{f}(y_{t,L})$.
This baseline learns an MLP $f$ and is trained to predict the target latent: $\lVert f(y_{t,L})-\overline{f}(y_{t,L})\rVert_{2}^{2}$.
The RND reward is calculated based on

$$ $\displaystyle R^{\text{RND}}=\frac{\lVert f(y_{t,L})-\overline{f}(y_{t,L})\rVert_{2}^{2}}{\text{std}\left(\lVert f(y_{t,L})-\overline{f}(y_{t,L})\rVert_{2}^{2}\right)}$ $$

The RND reward is then assigned to actions from trajectories that receive zero reward.

## Appendix B Additional Results

Impact of strategy prompt:
The strategy prompt is a short text asking the LLM to generate a strategy before generating an action.
The exact prompts for each environment are detailed in [Section˜A.1](#A1.SS1).
[Figure˜7](#A2.F7) compares standard GRPO training with and without the strategy prompt, demonstrating that the agent performs mostly the same with the strategy prompt and ends with slightly higher performance.
In this result, both approaches were allocated the same token budget for the per-action response.
From these results, all methods in [Section˜4](#S4) use the strategy prompt for a consistent comparison.

Impact of SGE on zero-shot pass@$k$: [Figure˜6](#A1.F6) shows how the mixed-temperature sampling in SGE affects the zero-shot performance of the model.
[Figure˜10](#A2.F10) breaks down the zero-shot performance across a variety of temperature settings in terms of the pass@1 performance.
The figure demonstrates that the mixed-temperature negatively impacts the pass@1.
However, this lower pass@1 comes with improved pass@16, as earlier shown in [Figure˜5(a)](#S4.F5.sf1), which ultimately leads to better RL.

Figure: Figure 7: Impact of prompting on RL training in Coding environment. Regular GRPO training is with 3 random seeds. Training without the strategy prompt is only with a single random seed.
Refer to caption: 2603.02045v1/x17.png

Qualitative strategy analysis: We provide qualitative examples of the strategies SGE produces during RL training in [Figure˜8](#A2.F8) and [Figure˜9](#A2.F9) for the coding environment. [Figure˜8](#A2.F8) shows an example of how SGE generates diverse strategies for the same problem. [Figure˜9](#A2.F9) shows an example of how SGE uses negative strategy reflection to improve the diversity of generated strategies and achieve successful outcomes in the environment.

State visitation exploration analysis: We further quantify the exploration capabilities of SGE by analyzing state-visitation of the method and regular GRPO training throughout the course of RL in the coding environment.
We do so by measuring the number of distinct outcomes encountered per task during RL training. For the coding environment, we count each distinct set of tests that fails as a distinct outcome.
This means we measure the functionality of the program rather than the code itself.
[Figure˜11](#A3.F11) shows that SGE encounters more diverse outcomes throughout RL training than regular GRPO.
This means that the programs produced by SGE better explore different functionalities in the Python execution environment.
This exploratory behavior helps SGE reach the sparse positive outcome more frequently during RL training.

Figure: Figure 8: Two strategies generated by SGE for the same task for the first step in the Coding environment. The text in “<…>" is for visualization and not part of the actual response. Where specified, some of the response is omitted for clarity. The strategies are under <Strategy 1>, <Strategy 2> and are independently generated and are only sequentially shown for visualization. The two strategies follow different approaches to solving the same task, with strategy 1 directly computing a frequency histogram via enumeration and strategy 2 using binary search and a sliding window. Each strategy is followed by by additional textual reasoning and Python code actually implementing the strategy.

Figure: Figure 9: Qualitative example in the Coding environment of how SGE uses negative strategy reflection to improve the diversity of generated strategies. The text in “<…>" is for visualization and not part of the actual response. Where specified, some of the response is omitted for clarity. The agent reflects on a previously executed strategy that resulted in a failed outcome from earlier in RL training with the negative strategy reflection prompt. The original strategy incorrectly missed a corner case by assuming the meeting point must be strictly after both positions. But this misses the case where the rightmost person’s building is itself a valid meeting point. The agent reflects on this strategy, sees the error, and then addresses the shortcomings of the old strategy with a new strategy.

Figure: Figure 10: Comparing the effect of the sampling temperature on the strategy and remaining tokens in terms of pass@1. Green indicates higher performance.
Refer to caption: 2603.02045v1/x18.png

## Appendix C Environment Details

### C.1 Coding

We turn the dataset of coding problems from into a multi-turn coding environment.
This dataset consists of Python coding problems categorized into “Easy”, “Medium”, and “Hard” difficulties.
For all of our training, we only work with the 606 problems from the “Hard” category.
We report test performance on the 228 hard category tasks from the test set with 8 independent evaluations per episode.

Each problem consists of a problem description, example input and outputs, constraints on input values, and the starter code.
The action space is a Python program, so the agent outputs self-contained Python code to solve the coding problem.
This code is run in a new Python instance with the test conditions provided by the problem episode and must finish within 2 seconds before timing out.
The reward is $+1$ if all of the tests pass within the time limit and 0 otherwise.
The observation for the next step is the stack trace of any runtime errors that occurred or any of the unit tests that failed.
In a multi-turn fashion, the agent generates a new program based on this observation.
For our experiments, all episodes last for 2 steps, meaning the agent has one action to correct errors from the first action.