## Contents
- 1 Introduction
  - Main finding.
  - Contributions.
- 2 Setup and Methods
  - Model and dataset.
  - Chain-of-thought generation.
  - Step-level entropy measurement.
  - Scalar coherence.
  - Binary monotonicity.
  - Step-level calibration.
  - Selective prediction.
- 3 Step-Level Calibration
  - ECE increases monotonically with step depth.
  - Token log-prob proxies share ranking but compress confidence range.
  - Answer-distribution confidence.
  - Interpretation.
- 4 Entropy Trajectory Dynamics
  - Scalar coherence is not predictive.
  - Binary monotonicity is a strong predictor.
  - Cross-model replication on Mistral-7B.
  - Shape-over-magnitude dissociation.
  - Failure mode analysis.
  - Qualitative examples.
  - Selective prediction and baseline comparison.
  - Violation count as a graded signal.
  - Token budget and equal-budget self-consistency.
  - Equal-budget fairness clarification.
  - Early-step monotonicity for compute-aware triage.
- 5 Related Work
  - Step-level signals for CoT reasoning.
  - Self-consistency and sampling-based reliability.
  - Single-trajectory reliability signals.
  - Temporal confidence dynamics.
  - Process reward models.
  - LLM calibration.
- 6 Conclusion
  - Limitations.
  - Future work.
- Appendix A Appendix
  - A.1 Ablation over Sampling Temperature τ \tau
  - A.2 Ablation over Number of Per-Step Samples m m
  - A.3 Implementation Details
    - Model.
    - Step segmentation.
    - Entropy computation.
    - Per-step completion format.
  - A.4 Early-Step Monotonicity (Prefix Analysis)
  - A.5 Full Confidence Proxy Results by Step Position
  - A.6 Monotonicity by Chain Length
  - A.7 Confounder Control: Partial Correlation and Logistic Regression
    - Partial correlation.
    - Logistic regression.
  - A.8 Difficulty-Proxy Control
  - A.9 ε \varepsilon -Tolerance Ablation
  - A.10 Step Exclusion Statistics
  - A.11 Equivalence of Entropy Monotonicity and Majority-Vote-Rate Monotonicity
  - A.12 Violation Count: Graded Shape Analysis
  - A.13 Segmentation Robustness
  - A.14 Equal-Mass ECE Binning
  - A.15 Small-Sample Entropy Bias and Miller–Madow Correction
  - A.16 Generalization to MATH Benchmark
  - A.17 Cross-Model Replication on GSM8K
  - A.18 Multi-Seed Stability Check
  - A.19 Empirical Equal-Budget SC/ESC Baselines
  - A.20 Coverage-Aware Self-Consistency Curves
  - A.21 Single-Trajectory Self-Judgment Baseline
    - Protocol details (reproducible).
    - Evaluation alignment.
    - Strict Yes/No self-evaluation baseline.
  - A.22 Token Accounting Details and Fairness

## Abstract

Abstract Chain-of-thought (CoT) reasoning improves LLM accuracy on complex tasks,
yet reliable methods for detecting reasoning failures without expensive multi-sample
approaches remain elusive. We study whether the shape of uncertainty dynamics
across reasoning steps—captured cheaply by sampling a handful of answer completions
at each step—predicts whether the final answer is correct. We introduce the concept of entropy-trajectory monotonicity : a chain is
monotone if its per-step answer-distribution entropy decreases at every step,
reflecting consistent uncertainty reduction. On GSM8K ( n = 300 n{=}300 ) with
Qwen2.5-7B-Instruct, monotone chains achieve 68.8 % 68.8\% accuracy versus 46.8 % 46.8\% for non-monotone chains—a gap of + 21.9 +21.9 percentage points (Fisher’s exact p = 0.0005 p{=}0.0005 ; OR = 2.50 =2.50 ). Critically, the scalar total entropy reduction
is not predictive ( ρ = − 0.06 \rho{=}{-}0.06 , p = 0.31 p{=}0.31 ), revealing a shape-over-magnitude dissociation : it is whether entropy decreases
at every step, not how much it drops, that predicts correctness.
The dissociation extends to the graded signal: in full-scale runs, increasing
violation count consistently reduces accuracy on both GSM8K and MATH-500. Beyond the 300-problem pilot, the effect persists at larger scale. On full
GSM8K ( n = 1319 n{=}1319 ), monotone chains reach 93.2 % 93.2\% accuracy versus 81.7 % 81.7\% for
non-monotone chains ( + 11.5 +11.5 pp). On MATH-500 ( n = 500 n{=}500 ), monotone chains reach 63.7 % 63.7\% versus 30.4 % 30.4\% ( + 33.3 +33.3 pp). Across both datasets, violation count is
negatively correlated with correctness (Spearman ρ = − 0.198 \rho=-0.198 on GSM8K and ρ = − 0.381 \rho=-0.381 on MATH-500). We further show that token log-probability confidence worsens in calibration
with step depth (ECE: 0.186 → 0.312 0.186\to 0.312 from step 0 to step 7), and that
entropy-trajectory monotonicity achieves + 5.8 +5.8 pp at 73.7 % 73.7\% coverage,
outperforming all scalar baselines including final-step entropy ( + 2.2 +2.2 pp) and
scalar coherence ( − 0.6 -0.6 pp, worse than random) at ≈ 1 , 500 \approx\!1{,}500 tokens/question—
one-eighth the cost of 40-chain self-consistency.
At matched answered-set coverage, SC@3/SC@5 can be slightly higher than
monotonicity ranking, so our claim is not dominance over SC voting, but a
cheap and interpretable single-chain triage signal with favorable
accuracy-cost trade-offs.
The initial pilot findings further replicate on a second model
(Mistral-7B-Instruct-v0.3, n = 300 n{=}300 ), where
monotone chains reach 72.3 % 72.3\% vs. 37.6 % 37.6\% for non-monotone chains
( + 34.7 +34.7 pp; OR = 4.33 =4.33 ). Structural properties of uncertainty trajectories are
thus more informative than aggregate magnitude measures across model families.
Current evidence is strongest on numeric/discrete-answer tasks; extending to
open-domain free-form QA requires stronger semantic canonicalization.

## 1 Introduction

Large language models (LLMs) produce correct and incorrect answers with equal
surface fluency. In chain-of-thought (CoT) reasoning
(Wei et al., 2022; Kojima et al., 2022), a model generates a multi-step solution
before giving a final answer—but a confident-looking chain of steps offers no
guarantee of correctness. Detecting failures cheaply, without generating many
independent samples, is an open practical problem.

Two families of reliability signals have been studied. *Self-consistency*
methods (Wang et al., 2023) aggregate answers from multiple independently sampled
chains, but require 10–40 samples per question and discard the rich step-level
information within each chain. *Token log-probability* scores are cheap but
systematically miscalibrated: we show that the ECE of token log-prob confidence
increases from $0.186$ at the first reasoning step to $0.312$ at the eighth
step ([Figure˜1](#S1.F1)a), with all standard proxy functions collapsing to
near-degenerate confidence ranges. Because step-level correctness labels are
unavailable in this black-box setup, this ECE trend is interpreted as
predictive calibration for final correctness rather than step-local calibration.

We pursue a different approach: rather than scoring a chain by a single scalar
derived from token probabilities, we ask how the model’s uncertainty over the
*final answer* evolves step by step. At each step prefix, we sample $m{=}5$
answer completions and compute the Shannon entropy $H_{k}$ of the resulting answer
distribution. This gives an *entropy trajectory* $(H_{0},H_{1},\ldots,H_{N})$
at low additional cost.

The central question we study is: does the *shape* of this trajectory
predict correctness, beyond what its *magnitude* can explain?

#### Main finding.

On GSM8K ($n{=}300$; Qwen2.5-7B-Instruct), chains whose entropy trajectory is
monotonically decreasing achieve $68.8\%$ accuracy, versus $46.8\%$ for
non-monotone chains—a gap of $+21.9$ pp (95% CI: [$+9.5$, $+34.3$];
Fisher’s exact $p{=}0.0005$; OR$=2.50$; [Figure˜1](#S1.F1)b). In contrast, the
total entropy drop $H_{0}{-}H_{N}$ (scalar coherence) has near-zero correlation
with correctness (Spearman $\rho{=}{-}0.06$, $p{=}0.31$). Whether uncertainty
decreases *consistently*—not how much it drops overall—is the
operative signal.
This pattern remains visible at larger scale: on full GSM8K ($n{=}1319$),
monotone chains achieve $93.2\%$ accuracy versus $81.7\%$ for non-monotone
chains ($+11.5$ pp), and on MATH-500 ($n{=}500$), $63.7\%$ versus $30.4\%$
($+33.3$ pp; [Section˜A.16](#A1.SS16)).

Figure: (a) ECE increases monotonically with step depth (0.186 at step 0 to 0.312 at step 7). Error bars show 95% bootstrap CIs.
Refer to caption: 2603.18940v2/x1.png

#### Contributions.

This paper is a *diagnostic study*: we introduce a new signal, establish
its empirical properties on GSM8K with Qwen2.5-7B, and characterize its failure
modes. We further replicate the effect on a second model family
(Mistral-7B-Instruct-v0.3) and provide an initial benchmark extension to MATH
([Sections˜A.16](#A1.SS16) and [A.17](#A1.SS17)).
Concretely, we contribute:

- 1.
Shape-over-magnitude dissociation. Binary entropy-trajectory
monotonicity is a strong predictor (OR$=2.5$, $p{=}0.0005$), while the
scalar total entropy drop is not ($\rho{=}{-}0.06$).
The dissociation extends to a graded signal: violation count
is negatively correlated with correctness on both full GSM8K
($\rho=-0.198$) and MATH-500 ($\rho=-0.381$), while violation magnitude is
unpredictive.
- 2.
Step-level calibration worsens with depth. Token log-prob ECE
rises from $0.186$ at step 0 to $0.312$ at step 7—the reverse of what one
might expect as the model approaches its final answer.
- 3.
Selective prediction vs. cheap baselines. Entropy-trajectory
monotonicity achieves $+5.8$ pp at $73.7\%$ coverage, outperforming all
scalar baselines (final-step entropy $+2.2$ pp; scalar coherence $-0.6$ pp,
worse than random; chain length $+2.6$ pp), at
$\approx\!1{,}500$ tokens/question. At matched answered-set coverage,
SC@3/SC@5 can be slightly higher, so we position monotonicity as a
compute-efficient, interpretable triage signal rather than a universal
replacement for SC voting.
- 4.
Extensive ablations and robustness checks.
Results are stable across $m\in\{3,5,10\}$ samples/step (gap
variation $<1.5$ pp), $\varepsilon\in[0,0.10]$ ($+21.9$ pp unchanged),
sampling temperature $\tau\in\{0.3,0.5,0.7,1.0\}$ (gap range $+14.4$–$+23.1$ pp, all substantially positive), and are confirmed
after Miller–Madow bias correction and confounder control
([Sections˜A.2](#A1.SS2), [A.1](#A1.SS1), [A.9](#A1.SS9), [A.15](#A1.SS15) and [A.7](#A1.SS7)).
The result is also robust to step-segmentation: restricting to the $96.7\%$
of chains with $N\geq 3$ steps yields an identical $+21.9$ pp gap
([Section˜A.13](#A1.SS13)).
The step-depth ECE trend is confirmed under equal-mass binning
([Section˜A.14](#A1.SS14)).

The remainder of this paper is organized as follows.
[Section˜2](#S2) describes our experimental setup and defines the key metrics.
[Section˜3](#S3) presents step-level calibration results.
[Section˜4](#S4) presents the entropy trajectory analysis.
[Section˜5](#S5) situates our work in the literature.
[Section˜6](#S6) concludes.

## 2 Setup and Methods

#### Model and dataset.

We use Qwen2.5-7B-Instruct (Team, 2024) with the model’s native chat
template applied via tokenizer.apply_chat_template—a detail that
matters: using raw text prompting instead reduces GSM8K accuracy from $\sim 63\%$
to $\sim 38\%$ on this model. We evaluate on a random sample of $n{=}300$
problems from the GSM8K test split (Cobbe et al., 2021), a standard benchmark
of grade-school arithmetic word problems requiring multi-step reasoning.

#### Chain-of-thought generation.

For each problem, we generate one CoT chain (temperature $\tau{=}0.1$,
max 512 tokens). The chain is segmented into steps by matching patterns of the
form “Step $k$:” in the generated text, with a sentence/newline fallback for
problems that do not produce explicit step markers. Token log-probabilities are
recorded at generation time for calibration analysis.

#### Step-level entropy measurement.

At each step prefix $s_{0}s_{1}\cdots s_{k}$, we sample $m{=}5$ answer completions
(temperature $\tau{=}0.7$, max 150 tokens) and extract final numerical answers.
The temperature $\tau{=}0.7$ introduces diversity in completions so that
entropy $H_{k}>0$ when the model is uncertain; lower temperatures would
underestimate uncertainty, and higher temperatures add noise unrelated to the
model’s actual uncertainty.
Ablations over $m\in\{3,5,10\}$ and $\tau\in\{0.3,0.5,0.7,1.0\}$ are
reported in [Sections˜A.2](#A1.SS2) and [A.1](#A1.SS1).
The per-step answer entropy is:

$$ $H_{k}=-\sum_{a\in\mathcal{A}}\hat{p}_{k}(a)\log\hat{p}_{k}(a),$ (1) $$

where $\hat{p}_{k}(a)$ is the empirical frequency of answer $a$ in the $m$ samples
at step $k$. This gives an entropy trajectory $(H_{0},H_{1},\ldots,H_{N})$ for each
chain of $N$ steps.

#### Scalar coherence.

The scalar coherence score is $\mathcal{C}=H_{0}-H_{N}$: the total entropy drop from
the start to the end of the chain. A high value indicates that, overall, the
model became more certain about the answer across the reasoning chain.

#### Binary monotonicity.

A chain is *$\varepsilon$-monotone* if its entropy trajectory decreases at every step up to a small tolerance:

$$ $H_{k+1}\leq H_{k}+\varepsilon\quad\text{for all }k\in\{0,\ldots,N{-}1\},$ (2) $$

where $\varepsilon\geq 0$ controls sensitivity to negligible fluctuations. We use $\varepsilon{=}0.01$ as the primary threshold; this is a per-step additive tolerance on entropy (measured in nats), not a cumulative bound. A chain is *non-monotone* if any single step violates [Equation˜2](#S2.E2). Strict monotonicity ($\varepsilon{=}0$) is the limiting case.

#### Step-level calibration.

Token log-probabilities are aggregated per step as the mean log-probability
$\bar{\ell}_{k}$ of tokens in step $k$. We compare four confidence proxy functions
mapping $\bar{\ell}_{k}$ to $[0,1]$: sigmoid-shifted ($\sigma(\bar{\ell}+1.5)$),
sigmoid-unshifted ($\sigma(\bar{\ell})$), raw-logprob (linear
normalization), and $\exp(\bar{\ell})$. For each proxy, we compute the Expected
Calibration Error (ECE) (Naeini et al., 2015; Guo et al., 2017) at each
step position $k$ using 10 equal-width bins, with 95% bootstrap confidence
intervals over $B{=}500$ resamples.

#### Selective prediction.

We evaluate the practical utility of the monotonicity signal by using it as a
coverage filter: the model “answers” only the subset of problems whose chain is
monotone and abstains on the rest. We report accuracy at this coverage level and
plot accuracy as a function of coverage under two ranking strategies: (1) sorting
by binary monotonicity first, then by scalar coherence; (2) sorting by scalar
coherence alone.

## 3 Step-Level Calibration

#### ECE increases monotonically with step depth.

[Table˜1](#S3.T1) summarizes four token log-probability confidence proxies and
the answer-distribution confidence proxy at step position $k{=}0$.
[Figure˜1](#S1.F1)a shows the ECE of the sigmoid-shifted proxy at each step
position $k{=}0,\ldots,7$ (positions with fewer than 10 observations are
excluded). ECE rises from $0.186$ at step 0 to $0.312$ at step 7—a $68\%$
relative increase. The trend is monotone across all eight positions,
though a formal Spearman test on $n{=}8$ step bins is underpowered
($\rho{=}0.27$, $p{=}0.45$).
Equal-mass (decile) binning confirms the trend, with ECE values
$0.195\to 0.357$ (step 0$\to$7); see [Section˜A.14](#A1.SS14) for a
step-by-step comparison.

**Table 1: Confidence proxies for step-0 correctness prediction on the pilot GSM8K split ($n{=}300$). ECE: Expected Calibration Error; AUROC: area under ROC; $\rho$: Spearman correlation with final answer correctness. All token log-probability proxies produce identical Spearman rankings and collapsed confidence ranges. Larger-scale GSM8K and MATH-500 robustness results are reported in [Sections˜4](#S4) and [A.16](#A1.SS16).**
| Confidence Proxy | ECE $\downarrow$ | AUROC $\uparrow$ | $\rho$ | Conf. Range |
| --- | --- | --- | --- | --- |
| $\sigma(\bar{\ell}+1.5)$ (sigmoid-shifted) | 0.186 | — | $+0.166^{**}$ | $[0.802,0.818]$ |
| $\sigma(\bar{\ell})$ (sigmoid-unshifted) | 0.133 | — | $+0.166^{**}$ | $[0.475,0.500]$ |
| $\bar{\ell}$ (raw-logprob, normalized) | 0.265 | — | $+0.166^{**}$ | $[0.990,1.000]$ |
| $\exp(\bar{\ell})$ (exp-logprob) | 0.254 | — | $+0.166^{**}$ | $[0.906,1.000]$ |
| Majority-vote rate (answer-dist) | 0.251 | 0.547 | $+0.084$ | $[0.144,0.749]$ |

#### Token log-prob proxies share ranking but compress confidence range.

All four token log-probability proxies produce identical Spearman correlations
with correctness ($\rho{=}+0.166$, $p{=}0.004$), because they are all strictly
monotone transformations of the same underlying log-probability values—they
yield the same ranking over problems. More importantly, their confidence
*ranges* collapse due to the normalization choices: the
sigmoid-shifted proxy assigns all 300 problems confidences in $[0.802,0.818]$,
a range of only $0.016$, while the raw-logprob proxy collapses to
$[0.990,1.000]$.
This range compression is a property of the chosen transformations combined with
the token-probability regime, not an intrinsic flaw of log-probs as a signal.
The practical consequence is that ECE metrics reflect the mean confidence vs. empirical accuracy gap rather than discrimination failure: models assign
token-level log-probabilities in a narrow regime that maps to
above $0.8$ after standard transformations, regardless of whether the answer is correct.

#### Answer-distribution confidence.

The majority-vote rate computed from the $m{=}5$ per-step samples (answer-dist
proxy) avoids this structural collapse, with confidence values spanning
$[0.14,0.75]$. However, its predictive power is weak: ECE$=0.251$,
AUROC$=0.547$, Spearman $\rho{=}+0.084$ ($p{=}0.15$, not significant). The
answer-distribution *scalar* at step 0 provides limited calibration.

#### Interpretation.

The step-level ECE trend suggests that token log-probability confidence—already
poorly calibrated at step 0—becomes progressively worse as reasoning
progresses. This is counterintuitive: one might expect a model to become
*better* calibrated as it approaches the final answer. The ECE increase
is driven by models becoming effectively more overconfident at later steps
(the mean confidence remains nearly constant while empirical accuracy varies),
not by any increase in probability variance. We interpret this as a descriptive
observation requiring further investigation across models and datasets.

A natural question is whether the answer-distribution at step $k$ is better
calibrated to *step-$k$ intermediate correctness* than to final-answer
correctness. This would require step-level correctness labels, which are
unavailable in our unsupervised setup; assigning them requires a process reward
model (Lightman et al., 2023) or human annotation. We therefore use
final-answer correctness as the only available calibration target, which
we acknowledge is a mismatch for early steps. Under this framing, our ECE
curves should be interpreted as *predictive calibration* of eventual
success, not as direct evidence about step-local correctness calibration.
The fact that the
answer-distribution scalar $\text{MVR}_{k}$ at step 0 achieves only
AUROC$=0.547$ for final correctness confirms that single-step scalars provide
limited calibration signal—motivating the trajectory-shape analysis of
[Section˜4](#S4).

## 4 Entropy Trajectory Dynamics

#### Scalar coherence is not predictive.

The scalar coherence $\mathcal{C}=H_{0}-H_{N}$ (total entropy drop) has Spearman
$\rho{=}{-}0.059$ with final answer correctness ($p{=}0.31$; not significant).
Correct answers are associated with *lower* total entropy reduction than
incorrect ones (mean coherence: $0.59$ correct vs. $0.68$ incorrect;
gap$={-}0.088$). This counterintuitive direction reflects cases where incorrect
chains converge quickly to a wrong answer—showing a large apparent coherence—
while correct chains may deliberate longer with more moderate convergence.

#### Binary monotonicity is a strong predictor.

Among the 300 evaluated chains, 221 ($73.7\%$) are monotone by the criterion in
[Equation˜2](#S2.E2). [Table˜2](#S4.T2) and [Figure˜1](#S1.F1)b show the main result:

**Table 2: Accuracy by entropy-trajectory monotonicity on the pilot GSM8K split ($n{=}300$, Qwen2.5-7B-Instruct). CI is 95% bootstrap; $p$ is Fisher’s exact test. Full-scale GSM8K ($n{=}1319$) and MATH-500 ($n{=}500$) robustness is reported in text.**
| Subset | $n$ | Accuracy | 95% CI | Fisher’s $p$ / OR |
| --- | --- | --- | --- | --- |
| All chains | 300 | $63.0\%$ | — | — |
| Monotone chains | 221 | $68.8\%$ | $[63.0,74.7]$ | $p{=}0.0005$ / OR$=2.50$ |
| Non-monotone chains | 79 | $46.8\%$ | $[35.7,57.9]$ | — |
| Accuracy gap (monotone $-$ non-monotone) | $+21.9$ pp | $[+9.5,+34.3]$ |  |  |

The accuracy gap of $+21.9$ pp is statistically robust (95% CI: [$+9.5$, $+34.3$]
via stratified bootstrap; Fisher’s exact $p{=}0.0005$; OR$=2.50$).

As an additional robustness check, a 3-seed sweep under the same
run-configuration yields consistently positive monotone/non-monotone gaps
(mean $+9.1$ pp; range $+5.2$ to $+13.8$ pp; details in
[Section˜A.18](#A1.SS18)).
Using a difficulty proxy control that includes SC@3 agreement, chain length,
and question length, monotonicity remains an independent positive predictor
(OR $\approx 2.37$; [Section˜A.8](#A1.SS8)).

At larger scale, the directional effect remains strong.
On full GSM8K ($n{=}1319$), monotone chains reach $93.2\%$ accuracy versus
$81.7\%$ for non-monotone chains (gap $+11.5$ pp) at monotone coverage
$68.1\%$. On MATH-500 ($n{=}500$), monotone chains reach $63.7\%$ versus
$30.4\%$ (gap $+33.3$ pp) at monotone coverage $27.0\%$
(details in [Section˜A.16](#A1.SS16)).

#### Cross-model replication on Mistral-7B.

We replicate the GSM8K experiment with Mistral-7B-Instruct-v0.3 ($n{=}300$,
same sampling setup and seed). The shape signal remains strong and significant:
monotone chains achieve $72.3\%$ accuracy vs. $37.6\%$ for non-monotone chains,
for a $+34.7$ pp gap (OR$=4.33$, Fisher’s $p{<}10^{-8}$), at monotone coverage
$39.7\%$ (details in [Section˜A.17](#A1.SS17)). While absolute accuracies differ
across model families, the core shape-over-magnitude effect persists.

#### Shape-over-magnitude dissociation.

The contrast between the null scalar result ($\rho{=}{-}0.06$) and the
significant binary result (OR$=2.50$, $p{=}0.0005$) reveals a
*shape-over-magnitude dissociation*: whether entropy decreases at every
step (shape) predicts correctness; how much it drops in total (magnitude) does
not.

The dissociation is conceptually informative. A chain that drops entropy sharply,
then rises on encountering a difficult sub-step, then drops again may exhibit a
large total coherence $\mathcal{C}$ while its non-monotone dynamics signal mid-chain
confusion. Conversely, a chain with a small but steady entropy decrease at every
step is monotone, indicating the model never “changes its mind” about the
answer direction.

#### Failure mode analysis.

Monotonicity is a diagnostic, not a perfect oracle.
Of the $221$ monotone chains, $69$ ($31.2\%$) are *false positives*: monotone
but incorrect. Of the $79$ non-monotone chains, $42$ ($53.2\%$) are *false
negatives*: non-monotone but correct.
As a classifier for final correctness, monotonicity has precision
$68.8\%$ (152/221), recall $80.4\%$ (152/189), and F1 $74.1\%$.
The signal is thus most useful as a triage filter—flagging likely-wrong answers
for re-sampling or abstention—rather than as a high-precision correctness
certificate. One natural extension is to combine monotonicity with complementary
signals (e.g., final-answer confidence from token log-probabilities or
majority-vote rate) in a lightweight scoring rule; such combinations could reduce
the false-positive rate at comparable coverage.

#### Qualitative examples.

[Figure˜2](#S4.F2) shows representative entropy trajectories. The
monotone chain shows a smooth decrease: each additional step reduces uncertainty.
The non-monotone chain shows an entropy spike at step 2—a point where the
model introduces an intermediate calculation that conflicts with its later
reasoning—before partially recovering.

Figure: Figure 2: Example per-step answer-distribution entropy trajectories. Left: A monotone trajectory (each step reduces $H_{k}$), corresponding to a correct final answer. Right: A non-monotone trajectory with a mid-chain entropy spike, corresponding to an incorrect answer. $H_{k}$ is defined in [Equation˜1](#S2.E1).
Refer to caption: 2603.18940v2/x3.png

#### Selective prediction and baseline comparison.

[Table˜3](#S4.T3) compares entropy-trajectory monotonicity against seven
cheap baselines at $73.7\%$ coverage.
Monotonicity achieves $+5.8$ pp over full coverage (AURC $=0.311$), outperforming
final-step entropy ($+2.2$ pp, AURC $=0.313$) and matching chain-length triage
on AURC (chain length: $+2.6$ pp, AURC $=0.310$).
The reviewer-requested single-trajectory baselines are weak in this setup:
self-judgment with one short verifier call reaches $62.4\%$ (AURC $=0.368$,
$\rho{=}+0.019$), and strict Yes/No self-evaluation reaches $63.3\%$
(AURC $=0.395$, $\rho{=}{-}0.019$), both well below trajectory-shape signals
(protocol in [Section˜A.21](#A1.SS21)).
Scalar coherence is also *worse than random* ($-0.6$ pp), confirming the
shape-over-magnitude dissociation.
[Figure˜3](#S4.F3) shows the full accuracy-coverage curves.

**Table 3: Cheap reliability signals on the pilot GSM8K split ($n{=}300$). Coverage fixed at $73.7\%$ ($k{=}221$). AURC: area under risk-coverage curve (lower is better; random $\approx 0.398$, oracle $=0.068$). ^∗: $p{<}0.05$; ^∗∗: $p{<}0.01$.**
| Signal | Acc@$73.7\%$ | $\Delta$ | Spearman $\rho$ | AURC |
| --- | --- | --- | --- | --- |
| Random | $62.9\%$ | $-0.1$ pp | $0.000$ | $0.398$ |
| Chain length (shorter first) | $65.6\%$ | $+2.6$ pp | $+0.101$ | $0.310$ |
| Final-step entropy $H_{N}$ | $65.2\%$ | $+2.2$ pp | $+0.093$ | $0.313$ |
| Scalar coherence $H_{0}{-}H_{N}$ | $62.4\%$ | $-0.6$ pp | $-0.059$ | $0.408$ |
| Self-judgment (1 verifier call) | $62.4\%$ | $-0.6$ pp | $+0.019$ | $0.368$ |
| Yes/No self-eval ($P(\text{Yes})$) | $63.3\%$ | $+0.3$ pp | $-0.019$ | $0.395$ |
| Max positive $\Delta H$ | $68.8\%$ | $+5.8$ pp | $+0.135^{*}$ | $0.340$ |
| Violation count ($-$vc) | $68.8\%$ | $+5.8$ pp | $+0.209^{**}$ | $\mathbf{0.311}$ |
| Entropy monotonicity (ours) | $\mathbf{68.8\%}$ | $\mathbf{+5.8}$ pp | $+0.200^{**}$ | $\mathbf{0.311}$ |

Figure: Figure 3: Accuracy vs. coverage for five cheap reliability signals. Entropy-trajectory monotonicity (solid blue) dominates all scalar baselines. Scalar coherence (dotted, AURC $=0.408$) is below the random baseline. Dashed line: full-coverage accuracy ($63.0\%$).
Refer to caption: 2603.18940v2/x4.png

#### Violation count as a graded signal.

The binary monotonicity flag (violation count $=0$) is a zero-threshold
discretization of the continuous violation count $v=\sum_{k}\mathbf{1}[H_{k+1}>H_{k}+\varepsilon]$. The graded signal reveals additional structure beyond the
binary split: chains with zero violations achieve $68.8\%$ accuracy, those with
exactly one violation achieve $50.8\%$, and those with two or more violations
achieve $28.6\%$ — a monotonically decreasing progression
([Section˜A.12](#A1.SS12)).
On full datasets, the same trend persists with larger sample sizes: on GSM8K,
accuracy drops from $93.2\%$ ($v{=}0$) to $86.4\%$ ($v{=}1$) to $72.3\%$
($v{=}2$), and on MATH-500 from $63.7\%$ to $43.4\%$ to $24.2\%$.
Correspondingly, Spearman correlation between violation count and correctness is
negative on both datasets ($\rho=-0.198$ on GSM8K, $\rho=-0.381$ on MATH-500).
Violation count is therefore a robust graded reliability indicator.
On the pilot GSM8K split, it also achieves the same AURC as binary monotonicity
(AURC $=0.311$). Crucially, the *magnitude* of violations does not add
predictive value: among non-monotone chains, the max positive $\Delta H$ has
Spearman $\rho{=}{-}0.017$ ($p{=}0.88$) with correctness, confirming that
*how many* disruptions occur matters, not *how large* they are.
This is a direct extension of the shape-over-magnitude dissociation to the
multi-violation setting.

#### Token budget and equal-budget self-consistency.

Our method uses $m\times\bar{N}\approx 5\times 4.9\approx 25$ short completions
(max 150 tokens) per question, totaling $\approx 1{,}500$ tokens—*two times cheaper*
than 10-chain self-consistency ($\approx 3{,}000$ tokens) and *eight times cheaper*
than 40-chain self-consistency ($\approx 12{,}000$ tokens).
We now run empirical equal-budget baselines on the same 300 GSM8K problems
([Table˜14](#A1.T14)). Under near-equal cost to our method,
SC@5 uses $1385.7$ tokens/problem on average and reaches $65.3\%$ accuracy.
SC@3 uses $831.4$ tokens/problem with $66.0\%$ accuracy.
An ESC-style early-stop simulation reaches $66.3\%$ accuracy at only
$673.6$ tokens/problem, stopping after 2.35 chains on average.
For reference, our monotonicity-gated policy yields $68.8\%$ accuracy on the
answered subset (73.7% coverage) with $\approx 1{,}500$ tokens/problem.
These empirical results replace the previous argument-only comparison and show
that full-chain voting improves over greedy decoding, while ESC offers a
stronger efficiency-accuracy trade-off among SC-style baselines.
To provide a matched-target comparison, we additionally compute
coverage-aware SC curves by ranking SC outputs with vote-agreement confidence.
At 73.7% coverage, SC@3 and SC@5 reach $70.6\%$ and $69.7\%$ answered-set
accuracy, respectively, compared with $68.8\%$ for our monotonicity ranking
(details in [Section˜A.20](#A1.SS20), including a unified matched-coverage
comparison figure).
This means monotonicity is not uniformly best on every selective metric;
its practical value is that it remains competitive while requiring only a
single primary chain plus short per-step probes, and simultaneously provides an
interpretable failure-detection signal (violation pattern) that scalar scores
do not recover.
To avoid metric mismatch, we report both coverage-aware and full-coverage views:
our $68.8\%$ number is answered-set accuracy at 73.7% coverage, whereas SC/ESC
numbers are full-coverage accuracies. Token accounting is likewise split into
(i) measured full-chain costs for SC/ESC and (ii) component-wise accounting for
our method (base-chain measured, per-step sampling term explicitly parameterized);
see [Section˜A.22](#A1.SS22) for the full breakdown and assumptions.

#### Equal-budget fairness clarification.

We compare two evaluation targets on purpose: selective performance for our
method (answered-set accuracy at fixed coverage) versus full-coverage
performance for SC/ESC (all-problem accuracy). The budgets are also tied to
different operational goals: our method spends tokens on per-step short
completions to rank and abstain, while SC/ESC spend tokens on additional
full-chain samples to vote before answering every problem. To provide a fair
empirical anchor rather than only theoretical scaling arguments, we report
executed SC@3, SC@5, and ESC-sim runs on the same 300-problem split with
measured token costs (
[Tables˜14](#A1.T14) and [A.22](#A1.SS22)).

#### Early-step monotonicity for compute-aware triage.

Prefix-only variants already provide useful separation. Using only the first
$k{=}2$ entropy transitions, prefix monotonicity achieves a $+16.7$ pp
accuracy gap (65.7% vs. 49.0%) on the $n{=}290$ problems with at least two
transitions, while using only $0.60\times$ the full trajectory cost on average
([Section˜A.4](#A1.SS4)). On the same subset, full-trajectory monotonicity gives
$+21.9$ pp, so the $k{=}2$ rule recovers about $76\%$ of the full gap at
substantially lower cost. This supports an early-exit deployment mode where
problems are triaged after the first two transitions and only ambiguous cases
are expanded to full trajectories.

## 5 Related Work

#### Step-level signals for CoT reasoning.

Recent work has begun exploring step-resolution information in CoT generation.
ConfSpec (Liu and He, 2026) uses per-step token confidence to trigger early
exit for speculative reasoning, accelerating inference without explicit
reliability modeling.
Concurrent work uses step-level confidence signals for RL fine-tuning of
reasoning chains (Cai and Sugiyama, 2026). Our work differs in goal: we study
uncertainty dynamics within a single chain for *diagnostic* purposes,
without any training.
Our work also differs in granularity from studies that aggregate across multiple
sampled CoT paths: we analyze trajectory shape *within* a single chain
rather than across parallel chains.

Active-Prompt (Diao et al., 2023) selects few-shot exemplars based on
final-answer disagreement/entropy over multiple sampled chains, showing that
answer-distribution entropy at the *output* is informative for difficulty
estimation. Our work complements this by studying entropy dynamics
*within* a single chain rather than across chains, and by showing that the
shape of the trajectory (not just the terminal value) carries diagnostic value.
Self-evaluation guided decoding (Xie et al., 2023) integrates step-wise
correctness signals into beam search; process reward models
(Lightman et al., 2023; Uesato et al., 2022) supervise reasoning at step resolution
using labeled data. In contrast, our method is fully unsupervised and requires
only inference, making it applicable without any labeled process data.

#### Self-consistency and sampling-based reliability.

The dominant approach to reliability estimation is self-consistency
(Wang et al., 2023): generating multiple independent CoT chains and taking
the majority answer. Extensions include complexity-based weighting
(Fu et al., 2023) and universal self-consistency
(Chen et al., 2023). These methods require 10–40 samples per question
and discard the step-level structure within each chain. Early-stopping
self-consistency (ESC; Li et al. 2024) reduces this cost by stopping
sampling once a majority is detected, but still operates at the level of full
chains and provides no step-level diagnostic. Our approach generates $m{=}5$
short completions per step (not full chains), providing step-resolution
information at ${\approx}1{,}500$ tokens per problem—about one-half the cost of
SC-10 and one-eighth of SC-40. The key distinction is that we use the
*trajectory shape* of within-chain sampling as a reliability signal,
rather than across-chain majority vote.

Xiong et al. (2026) addresses the complementary question of *when* to
invoke multi-path reasoning based on single-trajectory features, using learned
sampling controllers. Our method provides a cheap, unsupervised alternative
based purely on entropy dynamics, without requiring any training.

#### Single-trajectory reliability signals.

Several recent works propose lightweight single-chain reliability signals.
Xie et al. (2026) introduce anchor-token confidence: a model’s probability
on a single Yes/No self-evaluation token as a reliability indicator.
This signal is extremely cheap (one forward pass) but discards the
temporal structure of the reasoning chain.
Our entropy-trajectory monotonicity captures richer information by tracking
uncertainty *evolution* across steps, at the cost of $m{=}5$ short
completions per step.
Ghasemabadi and Niu (2025) study whether LLMs can predict their own failures
via self-judging, finding moderate success that complements structural signals.
White-box correctness verification (CRV; Zhao et al. 2026) exploits the
computation graph of the reasoning chain for verification, achieving higher
accuracy than sampling-based approaches but requiring access to model internals
and substantial compute.
Our method is fully black-box, requiring only the ability to generate
completions from a prefix.

#### Temporal confidence dynamics.

Several recent works explicitly model confidence as a temporal process.
Temporalizing Confidence (Mao et al., 2025) formulates stepwise
confidence trajectories with Signal Temporal Logic constraints (including
monotonicity-like templates), but operates on token-level confidence traces and
requires richer control over confidence shaping.
Recurrent Confidence Chain (Mao and Venkat, 2026) aggregates step confidence with a
temporal recurrent mechanism informed by inter-step dependencies, improving
calibration at the cost of additional modeling assumptions.
Thought Calibration (Wu et al., 2025) focuses on confidence-driven
test-time stopping with lightweight probes on hidden representations.
Compared with these methods, our approach is deliberately minimal: fully
unsupervised, black-box, and based only on sampled answer distributions from
prefix completions.

#### Process reward models.

Supervised process reward models (PRMs) assign credit to intermediate steps
using human annotations (Lightman et al., 2023) or automated annotations
(Uesato et al., 2022; Wang et al., 2024).
PRMs require training and labeled process data; our entropy-trajectory approach
is fully unsupervised and requires only inference. The monotonicity signal can
be viewed as a cheap proxy for process supervision: a chain that never increases
uncertainty about its answer is unlikely to contain a mid-chain error.

#### LLM calibration.

A substantial body of work studies the calibration of LLM predictions at the
output level (Guo et al., 2017; Kadavath et al., 2022; Xiong et al., 2024).
Kadavath et al. (2022) show that LLMs can express calibrated uncertainty
about factual knowledge through self-evaluation.
Xiong et al. (2024) survey calibration methods for LLMs, including
temperature scaling and verbal confidence elicitation.
Our work differs in studying how calibration evolves *within* a reasoning
chain—a temporal dimension absent from final-answer calibration studies—
and in observing that token log-probability confidence worsens with step depth.

## 6 Conclusion

We studied whether the shape of answer-distribution entropy dynamics across
reasoning steps predicts the correctness of chain-of-thought outputs.
On GSM8K with Qwen2.5-7B-Instruct, we found a clear shape-over-magnitude
dissociation: binary entropy-trajectory monotonicity is a significant predictor
of correctness (OR$=2.50$, Fisher’s $p{=}0.0005$, $+21.9$ pp accuracy gap),
while the scalar total entropy drop is not ($\rho{=}{-}0.06$, $p{=}0.31$).
Token log-probability confidence worsens in calibration from the first to the
last reasoning step, and monotonicity-based selective prediction achieves
$+5.8$ pp accuracy at $73.7\%$ coverage.
This directional signal also persists in larger runs: on full GSM8K
($n{=}1319$), monotone chains achieve $93.2\%$ vs. $81.7\%$ for non-monotone
chains ($+11.5$ pp), and on MATH-500 ($n{=}500$), $63.7\%$ vs. $30.4\%$
($+33.3$ pp).
The same directional effect replicates on Mistral-7B-Instruct-v0.3 with an even
larger gap ($+34.7$ pp; OR$=4.33$), supporting cross-family robustness.

#### Limitations.

Despite larger-scale runs and second-model replication, this study remains
limited in breadth.
MATH-500 ($n{=}500$) provides encouraging cross-dataset evidence within the math
domain (monotone $+$33.3 pp; [Section˜A.16](#A1.SS16)),
but broader task diversity is still needed.
On GSM8K, replication on Mistral-7B-Instruct-v0.3 confirms the core signal
([Section˜A.17](#A1.SS17)), but broader model coverage (e.g., Phi-3, Gemma,
Llama variants) is still needed.
Temperature ablations ($\tau\in\{0.3,0.5,0.7,1.0\}$) confirm that the
$+21.9$ pp gap is robust across sampling temperatures
([Section˜A.1](#A1.SS1)).
Threshold robustness and confounder control analyses ([Sections˜A.9](#A1.SS9) and [A.7](#A1.SS7))
show that the $+21.9$ pp gap is unchanged for all $\varepsilon\leq 0.10$ and
that partial correlation controlling for chain length remains significant
($r{=}0.179$, $p{=}0.0018$). Problem difficulty and other confounders have not
been controlled. The step-level ECE trend is a descriptive observation
underpowered for formal inference at $n{=}8$ step bins. The monotonicity signal
has a $31.2\%$ false-positive rate, limiting its precision as a standalone
correctness certificate; the graded violation count provides additional
resolution ([Section˜A.12](#A1.SS12)). The calibration analysis uses
final-answer correctness as the target because step-level correctness labels
are unavailable in our unsupervised setup.
Recent temporal-confidence methods that use stronger supervision or hidden-state
access (e.g., STL-style confidence shaping, recurrent confidence aggregation,
and thought-calibration probes) may achieve better calibration in white/gray-box
settings; integrating those ideas with black-box trajectory sampling is open.
Anchor-token confidence (Xie et al. 2026) and self-judgment
(Ghasemabadi and Niu, 2025) are not included as baselines in the current
study; comparing them directly requires implementing single-token self-evaluation
prompts on the same problems, which we defer to follow-up work.
Results are still concentrated in arithmetic and word-problem style reasoning.
The current HotpotQA run is not yet suitable for headline conclusions because
context integration and QA-specific evaluation need to be standardized before
cross-domain comparison.

#### Future work.

Immediate extensions include (1) replicating across additional models
(e.g., Llama-3-8B, Mistral) and datasets (MATH full, AQuA, HotpotQA);
(2) combining the monotonicity signal with complementary features (final-answer
confidence, violation count) in a learned scoring rule to reduce the
$31.2\%$ false-positive rate;
(3) using the monotonicity flag as a trigger for targeted re-sampling—
selectively applying multi-chain self-consistency only when a chain is
non-monotone—to reduce average token budget below full self-consistency; and
(4) studying adversarial scenarios where models are trained to enforce
monotonic entropy trajectories regardless of correctness.

## Appendix A Appendix

### A.1 Ablation over Sampling Temperature τ \tau

[Table˜4](#A1.T4) reports the effect of varying the per-step completion
sampling temperature $\tau\in\{0.3,0.5,0.7,1.0\}$ on monotonicity rate
and the accuracy gap ($m{=}5$, $n{=}300$, $\varepsilon{=}0.01$).
The $\tau{=}0.7$ row is the main result.

**Table 4: Effect of sampling temperature $\tau$ on the entropy-monotonicity signal. Lower $\tau$ underestimates uncertainty (more deterministic completions); higher $\tau$ adds noise unrelated to model uncertainty. Gap = monotone $-$ non-monotone accuracy.**
| $\tau$ | $n$ | Mono. rate | Mono. acc / Non-mono. acc | Gap |
| --- | --- | --- | --- | --- |
| $0.3$ | $300$ | $0.697$ | $0.703$ / $0.473$ | $+23.1$ pp |
| $0.5$ | $300$ | $0.707$ | $0.689$ / $0.523$ | $+16.6$ pp |
| $0.7$ | $300$ | $0.737$ | $0.688$ / $0.469$ | $+21.9$ pp |
| $1.0$ | $300$ | $0.723$ | $0.687$ / $0.542$ | $+14.4$ pp |

### A.2 Ablation over Number of Per-Step Samples m m

[Table˜5](#A1.T5) reports the effect of varying the number of per-step
completion samples $m\in\{3,5,10\}$ on monotonicity rate, per-group
accuracy, and the accuracy gap ($\varepsilon{=}0.01$). The $m{=}5$ row ($n{=}300$)
is the main result.

**Table 5: Effect of number of per-step samples $m$ on the entropy-monotonicity signal. Monotone and non-monotone accuracy are at $\varepsilon{=}0.01$. Gap = monotone $-$ non-monotone accuracy. The $m{=}10$ run used $n{=}212$ because the $m{=}3$ and $m{=}10$ experiments ran concurrently on the same GPU; occasional CUDA memory errors at long steps excluded some problems.**
| $m$ | $n$ | Mono. rate | Mono. acc / Non-mono. acc | Gap |
| --- | --- | --- | --- | --- |
| $3$ | $300$ | $0.843$ | $0.672$ / $0.447$ | $+22.5$ pp |
| $5$ | $300$ | $0.737$ | $0.688$ / $0.469$ | $+21.9$ pp |
| $10$ | $212$ | $0.698$ | $0.743$ / $0.531$ | $+21.2$ pp |

The accuracy gap is essentially identical across all three values: $+22.5$, $+21.9$,
and $+21.2$ pp for $m{=}3$, $5$, and $10$ respectively—a variation of less than
$1.5$ pp. As $m$ increases, the monotonicity rate decreases (more samples detect
more violations), but the gap between monotone and non-monotone accuracy remains
stable. This stability demonstrates that the choice of $m{=}5$ in the main
experiment is not critical, and the shape-over-magnitude dissociation is robust
to the precision of the entropy estimates.

### A.3 Implementation Details

#### Model.

Qwen2.5-7B-Instruct is loaded in bfloat16 precision using the
Hugging Face transformers library. The model’s native chat template
is applied via tokenizer.apply_chat_template with a chain-of-thought
system prompt. The VLLM backend is not used; standard autoregressive generation
is performed with model.generate.

#### Step segmentation.

Steps are extracted by matching the regular expression Step [0-9]+:
in the generated text; on match, the Step N: prefix is stripped and
the content stored. When no such pattern is found, the text is split on
double newlines or sentence boundaries. All stored step texts therefore contain
only the step content, not the numbering marker.
In our experiments, $96.7\%$ of chains have $N\geq 3$ steps, confirming
multi-step generation for nearly all chains; robustness of the main result
to step-count filtering is reported in [Section˜A.13](#A1.SS13).

#### Entropy computation.

At each step prefix, $m{=}5$ completions are sampled independently
(temperature $\tau{=}0.7$, max 150 tokens). Final numerical answers are
extracted using a regex matching integers and decimals at the end of each
completion. If no numerical answer is found, the completion is discarded;
steps where fewer than 2 completions produced parseable answers are excluded
from the trajectory.

#### Per-step completion format.

For each prefix, we continue generation from the prefix text directly (no
extra verifier prompt) and then parse the generated continuation for the final
numeric answer using the regex described above. This design keeps the pipeline
fully black-box and model-agnostic: it does not require model-internal logits
beyond normal decoding outputs, and it avoids introducing an additional
prompt-template confounder between step positions.

### A.4 Early-Step Monotonicity (Prefix Analysis)

To test how early trajectory shape becomes useful, we recompute monotonicity
using only the first $k$ transitions of the entropy curve
($H_{0},\ldots,H_{k}$), with $k\in\{1,2,3\}$. Results are reported in
[Table˜6](#A1.T6). Prefix-$k$ compute is measured as the average
fraction of transitions evaluated relative to each chain’s full trajectory.

**Table 6: Prefix monotonicity on GSM8K from figures/prefix_results.json. Gap = monotone $-$ non-monotone accuracy. “Full (same $n$ as $k{=}2$)” controls for subset shift.**
| Prefix rule | $n$ | Coverage | Mono. acc | Non-mono. acc | Gap | Cost ratio |
| --- | --- | --- | --- | --- | --- | --- |
| $k{=}1$ | 300 | $91.0\%$ | $64.1\%$ | $51.9\%$ | $+12.3$ pp | $0.32$ |
| $k{=}2$ | 290 | $82.4\%$ | $65.7\%$ | $49.0\%$ | $+16.7$ pp | $0.60$ |
| $k{=}3$ | 225 | $72.4\%$ | $63.8\%$ | $50.0\%$ | $+13.8$ pp | $0.73$ |
| Full (all) | 300 | $73.7\%$ | $68.8\%$ | $46.8\%$ | $+21.9$ pp | $1.00$ |
| Full (same $n$ as $k{=}2$) | 290 | $72.8\%$ | $68.7\%$ | $46.8\%$ | $+21.9$ pp | $1.00$ |

The key compute-aware result is that $k{=}2$ already recovers most of the
signal: $+16.7$ pp at only $60\%$ of full trajectory cost. Relative to the
matched full-trajectory subset ($+21.9$ pp), this is about $76\%$ of the full
accuracy gap.

### A.5 Full Confidence Proxy Results by Step Position

[Table˜7](#A1.T7) gives the full ECE values and 95% bootstrap
confidence intervals for the sigmoid-shifted proxy at each step position.

**Table 7: ECE by step position (sigmoid-shifted proxy; 95% bootstrap CI, $B{=}500$ resamples). $n_{k}$ is the number of problems with at least $k$ steps.**
| Step $k$ | $n_{k}$ | ECE | 95% CI |
| --- | --- | --- | --- |
| 0 | 300 | 0.186 | — |
| 1 | 300 | 0.186 | — |
| 2 | 288 | 0.190 | — |
| 3 | 241 | 0.218 | — |
| 4 | 175 | 0.222 | — |
| 5 | 118 | 0.232 | — |
| 6 | 73 | 0.256 | — |
| 7 | 42 | 0.312 | — |

### A.6 Monotonicity by Chain Length

Longer chains have more opportunities for non-monotone steps. Among chains with
$N\leq 3$ steps, monotonicity rate is $81.2\%$; among chains with $N\geq 6$
steps, it is $64.7\%$ (Spearman $\rho{=}{-}0.37$ between chain length and
monotonicity flag, $p{<}0.0001$). Chain length is therefore a confounder to
control for.

### A.7 Confounder Control: Partial Correlation and Logistic Regression

To isolate the independent contribution of entropy-trajectory monotonicity from
chain-length and final-entropy confounders, we run two supplementary analyses
on the $n{=}300$ chains.

#### Partial correlation.

We residualize both the monotonicity flag and the correctness label on chain
length $N$ (linear regression), then compute the Pearson correlation of the
residuals. The partial correlation is $r{=}0.179$ ($p{=}0.0018$), confirming
that monotonicity predicts correctness even after removing the chain-length
component from both variables.

#### Logistic regression.

A logistic regression with correctness as the outcome and three predictors—
binary monotonicity, chain length $N$, and final entropy $H_{N}$ (all standardized)
—yields coefficients $\beta_{\text{mono}}{=}0.36$, $\beta_{N}{=}{-}0.05$,
$\beta_{H_{N}}{=}{-}0.07$, with AUROC$=0.627$. A monotonicity-only model achieves
AUROC$=0.591$; adding chain length and final entropy provides a modest improvement.
The positive coefficient on monotonicity is consistent with the main Fisher’s
exact result (OR$=2.50$).

### A.8 Difficulty-Proxy Control

To test whether monotonicity is only a proxy for item difficulty, we add a
stronger difficulty proxy based on low-$N$ self-consistency agreement.
Specifically, we fit a logistic model on the same 300 GSM8K items:

$$ $\text{correct}\sim\text{monotone}+z(\text{chain_len})+z(\text{question_len})+z(\text{SC@3 agreement}).$ $$

Results are computed by figures/compute_difficulty_control.py
and summarized in figures/difficulty_control.json.

The monotonicity coefficient remains positive and significant under this
control (coef $=0.861$, OR $=2.37$, bootstrap $p\approx 0.003$,
95% CI for OR $[1.43,3.98]$), while chain length and question length are
near-null in this specification. SC@3 agreement is also significant
(OR $=1.70$, $p\approx 0.003$), suggesting both signals contribute
complementary information.

**Table 8: Difficulty-proxy controlled logistic regression on GSM8K ($n{=}300$). Bootstrap-based uncertainty estimates are used for the fallback solver.**
| Variable | Coef. | Odds ratio | $p$-value |
| --- | --- | --- | --- |
| monotone | 0.861 | 2.367 | 0.003 |
| chain len z | 0.020 | 1.020 | 0.880 |
| question len z | 0.019 | 1.019 | 0.933 |
| sc3 agreement z | 0.528 | 1.696 | 0.003 |

As an additional stratified check, we group items by SC@3 agreement level
($1/3$, $2/3$, and $1$) and recompute the monotone/non-monotone gap within each
stratum. The gap remains positive in all three groups (from $+18.2$ pp to
$+23.5$ pp), with weighted average $+21.5$ pp. This supports the interpretation
that monotonicity is not reducible to a single difficulty proxy.

### A.9 ε \varepsilon -Tolerance Ablation

[Table˜9](#A1.T9) reports the monotonicity rate, monotone accuracy,
non-monotone accuracy, and the accuracy gap for $\varepsilon\in\{0.000,0.005,0.010,0.020,0.050,0.100,0.200\}$.

**Table 9: Sensitivity of the monotonicity result to $\varepsilon$. Accuracy gap = monotone accuracy $-$ non-monotone accuracy. Results are essentially identical across $\varepsilon\in[0,0.10]$, confirming that the choice $\varepsilon{=}0.01$ is not critical.**
| $\varepsilon$ | Mono. rate | Mono. acc | Non-mono. acc | Gap |
| --- | --- | --- | --- | --- |
| $0.000$ | $0.737$ | $0.688$ | $0.468$ | $+21.9$ pp |
| $0.005$ | $0.737$ | $0.688$ | $0.468$ | $+21.9$ pp |
| $0.010$ | $0.737$ | $0.688$ | $0.468$ | $+21.9$ pp |
| $0.020$ | $0.737$ | $0.688$ | $0.468$ | $+21.9$ pp |
| $0.050$ | $0.737$ | $0.688$ | $0.468$ | $+21.9$ pp |
| $0.100$ | $0.737$ | $0.688$ | $0.468$ | $+21.9$ pp |
| $0.200$ | $0.740$ | $0.685$ | $0.474$ | $+21.0$ pp |

The result is remarkably stable: the $+21.9$ pp gap is unchanged for all
$\varepsilon\leq 0.10$, and the gap shrinks to only $+21.0$ pp at $\varepsilon{=}0.20$.
This stability arises because entropy jumps in non-monotone chains tend to be
larger than $0.20$ nats; the $\varepsilon$ threshold only matters for very small
fluctuations, which are rare in practice.

### A.10 Step Exclusion Statistics

Steps with fewer than 2 parseable numerical answers are excluded from the entropy
trajectory (see [Section˜A.3](#A1.SS3)). In our $n{=}300$ evaluation, *zero steps*
were excluded: all 1,474 nominal steps across all chains produced at least 2
parseable answers. The step-exclusion mechanism therefore introduces no selection
bias in our results.

### A.11 Equivalence of Entropy Monotonicity and Majority-Vote-Rate Monotonicity

A reviewer asked whether *monotonicity of the majority-vote rate*—the
fraction of the $m$ completions that agree on the most common answer—would be
an equivalent or distinct signal.
The majority-vote rate $r_{k}=\max_{a}\hat{p}_{k}(a)$ is a different summary
statistic from the Shannon entropy $H_{k}=-\sum_{a}\hat{p}_{k}(a)\log\hat{p}_{k}(a)$,
so monotone entropy trajectories and monotone majority-vote-rate trajectories
are not identical in general.

In our $n{=}300$ evaluation, we computed both signals with $\varepsilon{=}0.01$
for entropy and a symmetric $0.05$ tolerance for MVR.
The two classifiers *agreed on 298 out of 300 chains ($99.3\%$)*, and
produced identical accuracy gaps (+21.9 pp). The two chains on which they
disagree involve entropy increases alongside majority-vote-rate increases, a
pattern that arises when competing wrong answers consolidate.
This near-perfect agreement confirms that the shape-over-magnitude dissociation
is not specific to Shannon entropy: any reasonable monotone measure of
distribution concentration yields the same qualitative result.

### A.12 Violation Count: Graded Shape Analysis

[Table˜10](#A1.T10) reports accuracy stratified by the number of
$\varepsilon$-violations $v=\sum_{k}\mathbf{1}[H_{k+1}>H_{k}+\varepsilon]$
in the entropy trajectory ($\varepsilon{=}0.01$) on full-scale GSM8K and
MATH-500 runs.

**Table 10: Accuracy by violation count on full GSM8K and MATH-500. Violation count $v$ equals the number of steps with $H_{k+1}>H_{k}+\varepsilon$ ($\varepsilon{=}0.01$). $v{=}0$ corresponds to monotone chains.**
| Dataset | Violation bucket | $n$ | Accuracy | Spearman $\rho(v,y)$ |
| --- | --- | --- | --- | --- |
| GSM8K ($n{=}1319$) | 0 | $898$ | $93.2\%$ | $-0.198$ |
|  | $1$ | $302$ | $86.4\%$ |  |
|  | $2$ | $83$ | $72.3\%$ |  |
|  | $\geq 3$ | $36$ | $63.9\%$ |  |
| MATH-500 ($n{=}500$) | 0 | $135$ | $63.7\%$ | $-0.381$ |
|  | $1$ | $173$ | $43.4\%$ |  |
|  | $2$ | $120$ | $24.2\%$ |  |
|  | $\geq 3$ | $72$ | $9.7\%$ |  |

The trend is monotone on both datasets: fewer violations imply higher
correctness, and the effect is stronger on MATH-500. This confirms that the
graded signal carries additional information beyond the binary split and scales
beyond the original 300-problem pilot. As in the main experiment, the magnitude
of violations does not add reliable predictive value among non-monotone chains
($\max_{k}\Delta H_{k}$ remains near-null), supporting a count-driven
shape-over-magnitude interpretation.

### A.13 Segmentation Robustness

Step segmentation uses a two-stage heuristic: first, split on the regex
Step [0-9]+:; if this yields $\leq 1$ part, fall back to sentence- or
newline-based splitting (see [Section˜A.3](#A1.SS3)).
To check whether results depend on the segmentation method, we restrict analysis
to the $96.7\%$ of chains with $N\geq 3$ steps—chains short enough to arise
from the fallback are excluded.

Among these 290 chains, the monotone group achieves $68.7\%$ accuracy and the
non-monotone group $46.8\%$—a gap of $+21.9$ pp (OR$=2.49$, Fisher’s exact
$p{=}0.0010$), matching the full-dataset result within rounding.
The main finding is therefore insensitive to the presence of a small number of
short (2-step) chains that may have used the fallback segmentation path.

### A.14 Equal-Mass ECE Binning

[Figure˜1(a)](#S1.F1.sf1) uses equal-width bins for the ECE computation.
[Table˜11](#A1.T11) compares equal-width and equal-mass (decile) binning
for each step position.

**Table 11: ECE by step position: equal-width (EW) vs. equal-mass (EM) binning. $n_{k}$ = problems with at least $k$ steps. Both binning strategies show the same increasing trend; equal-mass binning produces slightly higher ECE at later steps owing to heavier weighting of the tails.**
| Step $k$ | $n_{k}$ | ECE (EW) | ECE (EM) |
| --- | --- | --- | --- |
| 0 | 300 | 0.186 | 0.195 |
| 1 | 300 | 0.186 | 0.186 |
| 2 | 290 | 0.190 | 0.188 |
| 3 | 225 | 0.218 | 0.216 |
| 4 | 160 | 0.222 | 0.222 |
| 5 | 89 | 0.232 | 0.242 |
| 6 | 50 | 0.256 | 0.293 |
| 7 | 25 | 0.312 | 0.357 |

Both binning strategies confirm the step-depth miscalibration trend reported in
[Section˜3](#S3).
The ECE increase from step 0 to step 7 is $+0.126$ (EW) and $+0.162$ (EM),
in both cases more than doubling the initial miscalibration.
Note that $n_{k}$ drops sharply for $k\geq 6$, so these values carry wider
confidence intervals; the trend at steps 0–5 (where $n_{k}\geq 89$) is the most
reliable portion of the curve.

### A.15 Small-Sample Entropy Bias and Miller–Madow Correction

With $m{=}5$ completions per step, the empirical entropy estimator
$\hat{H}_{k}=-\sum_{a}\hat{p}_{k}(a)\log\hat{p}_{k}(a)$ is negatively biased.
The Miller–Madow correction adds $(K_{k}-1)/(2m)$ to each $\hat{H}_{k}$,
where $K_{k}$ is the number of unique observed answers at step $k$.
Since $K_{k}\leq m=5$, the correction is at most $0.4$ nats per step; for the
binary case ($K_{k}{=}2$), it equals $0.1$ nats.

The correction affects the *level* of each entropy estimate but not the
pairwise *differences* $H_{k}-H_{k+1}$ unless $K_{k}$ varies across steps.
We verified that among the 79 non-monotone chains, the minimum violation
magnitude is $0.12$ nats—larger than any possible single-step MM correction
when $K$ changes by at most 1 between adjacent steps. Furthermore, zero
borderline chains have violations $<0.1$ nats, so *no monotonicity
decision would be reversed* by applying the Miller–Madow correction.
The main results are therefore robust to small-sample entropy bias.

### A.16 Generalization to MATH Benchmark

To test whether the entropy-monotonicity signal generalizes beyond GSM8K, we
evaluate on MATH-500 ($n{=}500$), using the same Qwen2.5-7B-Instruct model and
$m{=}5$, $\tau{=}0.7$. Results are reported in [Table˜12](#A1.T12).
The signal generalizes strongly: monotone chains achieve $63.7\%$ accuracy vs. $30.4\%$ for non-monotone chains ($+33.3$ pp gap). Compared with full GSM8K
($n{=}1319$, gap $+11.5$ pp), MATH-500 has lower overall accuracy ($39.4\%$)
and lower monotone coverage ($27.0\%$), consistent with the task being more
difficult and answer distributions more diverse.

**Table 12: Entropy-trajectory monotonicity on full GSM8K and MATH-500, Qwen2.5-7B-Instruct. The monotone group is more accurate on both datasets, with a larger gap on MATH-500.**
| Dataset | $n$ | Mono. rate | Mono. acc / Non-mono. acc | Gap |
| --- | --- | --- | --- | --- |
| GSM8K (full) | $1319$ | $0.681$ | $0.932$ / $0.817$ | $+11.5$ pp |
| MATH-500 | $500$ | $0.270$ | $0.637$ / $0.304$ | $+33.3$ pp |

### A.17 Cross-Model Replication on GSM8K

To address the single-model concern directly, we replicate the main GSM8K setup
on Mistral-7B-Instruct-v0.3 with identical protocol
($n{=}300$, $m{=}5$, $\tau{=}0.7$, seed 42). Results are summarized in
[Table˜13](#A1.T13).

**Table 13: Cross-model replication on GSM8K. The monotonicity signal remains strong across model families, though coverage differs.**
| Model | $n$ | Mono. rate | Mono. acc | Non-mono. acc | Gap |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 300 | $0.737$ | $0.688$ | $0.468$ | $+21.9$ pp |
| Mistral-7B-Instruct-v0.3 | 300 | $0.397$ | $0.723$ | $0.376$ | $+34.7$ pp |

For Mistral, Fisher’s exact test gives OR$=4.33$ and
$p=2.66\times 10^{-9}$, with a bootstrap 95% CI of
[$+23.4$, $+45.3$] pp for the monotone/non-monotone accuracy gap.
This confirms that the signal is not specific to the Qwen model family.

### A.18 Multi-Seed Stability Check

To test seed sensitivity, we rerun GSM8K ($n{=}300$ each) with three additional
random seeds (11, 22, 33) and aggregate results with
figures/aggregate_seed_stability.py
(output: figures/seed_stability.json).
Across seeds, the monotone/non-monotone accuracy gap remains positive:
$+8.3$ pp (seed 11), $+13.8$ pp (seed 22), and $+5.2$ pp (seed 33), for a mean
of $+9.1$ pp (std $4.3$). Smoothed odds ratios are also consistently above 1
(mean $3.24$).

Absolute accuracies in this sweep are higher than in the main seed-42 run,
indicating that these runs should be interpreted as a directional stability
check rather than a direct replacement of the main-table operating point.

Figure: Figure 4: Monotone $-$ non-monotone accuracy gap across three additional seeds on GSM8K ($n{=}300$ per seed). The gap remains positive for all seeds.
Refer to caption: 2603.18940v2/x5.png

### A.19 Empirical Equal-Budget SC/ESC Baselines

To address the reviewer request for *executed* equal-budget baselines, we
run full-chain self-consistency and ESC-style stopping on the same 300 GSM8K
problems with Qwen2.5-7B-Instruct (seed 42). Results are from
sc_baseline/results.json.

**Table 14: Empirical SC/ESC comparison on GSM8K ($n{=}300$). “Our selective” uses entropy-trajectory monotonicity and reports answered-set accuracy at 73.7% coverage (main result). SC/ESC report full-coverage accuracy.**
| Method | Accuracy | Avg tokens / problem | Coverage |
| --- | --- | --- | --- |
| SC@3 | $66.0\%$ | $831.4$ | $100\%$ |
| SC@5 (near-equal budget) | $65.3\%$ | $1385.7$ | $100\%$ |
| ESC-sim (min 2 chains) | $66.3\%$ | $673.6$ | $100\%$ |
| Our selective (monotonicity) | $68.8\%$ | $\approx 1500$ | $73.7\%$ |

ESC stops at 2 chains for 234/300 problems (78.0%), at 3 chains for 46/300
(15.3%), and uses all 5 chains for 20/300 (6.7%), yielding an average stop
point of 2.35 chains. This confirms the expected efficiency advantage of ESC
in full-coverage operation.

### A.20 Coverage-Aware Self-Consistency Curves

To match the selective-prediction target directly, we build coverage-aware SC
rankings from the same sc_baseline/per_problem.json: for each
problem, SC confidence is defined as vote agreement (majority-vote fraction)
among the sampled full-chain answers, and problems are ranked by this score.
We compute curves for both SC@3 and SC@5 and evaluate answered-set accuracy as
coverage increases. Results are in
figures/sc_coverage_aware.json; plotting code is
figures/gen_fig_sc_coverage_aware.py.

At the same operating coverage as our method (73.7%, $k{=}221$),
coverage-aware SC@3 reaches $70.6\%$ and SC@5 reaches $69.7\%$, while our
monotonicity-first ranking is $68.8\%$. This narrows and reverses the gap seen
in full-coverage numbers, and clarifies that comparisons are sensitive to
whether methods are evaluated as full-coverage voters or selective triage
policies.

For a broader matched-coverage head-to-head, we additionally include
self-judgment and strict Yes/No self-evaluation curves in
figures/fig5_matched_coverage_comparison.pdf
(summary: figures/matched_coverage_summary.json).

Figure: Figure 5: Coverage-aware answered-set accuracy on GSM8K ($n{=}300$). SC confidence is vote-agreement fraction for SC@3/SC@5; our curve uses monotonicity-first ranking. Dotted line marks 73.7% coverage.
Refer to caption: 2603.18940v2/x6.png

Figure: Figure 6: Unified matched-coverage comparison on GSM8K ($n{=}300$): our monotonicity-first ranking, SC@3/SC@5 vote-agreement rankings, self-judgment confidence, and strict Yes/No self-evaluation confidence. All methods are evaluated as answered-set accuracy versus coverage.
Refer to caption: 2603.18940v2/x7.png

### A.21 Single-Trajectory Self-Judgment Baseline

To address the reviewer request for a strong single-trajectory comparator, we
run a self-judgment baseline on the same 300 GSM8K problems using
Qwen2.5-7B-Instruct. For each problem, the model receives its own final answer
and emits one short JSON confidence score in $[0,100]$ via a verifier prompt
(one generation call per problem). We then rank by this confidence and report
the same selective metrics at target coverage $0.737$. Results are from
figures/self_judgment_baseline.json.

#### Protocol details (reproducible).

Implementation is in figures/run_self_judgment_baseline.py.
For each item, we construct a two-message chat prompt:
system: “You are a strict math-answer verifier… output exactly one
line JSON {c̈onfidence:̈ <0–100>}.”
user: “Question: … Proposed final answer: … Return only JSON.”
Decoding uses one verifier generation with max_new_tokens=64,
temperature $0.0$ (greedy), and the model’s native chat template.
Confidence extraction follows a deterministic rule: parse
"confidence": number if present; else parse the first numeric token
in output; else map lexical cues (“yes/correct” $\rightarrow 0.75$,
“no/incorrect” $\rightarrow 0.25$), with a final fallback $0.5$.

#### Evaluation alignment.

To match our selective setup, self-judgment produces one score per problem,
then ranks all 300 items by score and reports Acc@coverage at
$k{=}\mathrm{round}(0.737\times 300)=221$ retained items. We also report
Spearman $\rho$ (score vs.
final correctness) and AURC under the same
risk-coverage definition used in the main text (lower is better).

Self-judgment yields Spearman $\rho{=}+0.019$, Acc@$73.7\%{=}62.4\%$, and
AURC $=0.368$, with mean verifier cost 7.6 tokens/problem (total
$242.3$ tokens/problem including the base chain). Despite low token overhead,
the ranking quality is weak and does not improve over random triage at matched
coverage, remaining substantially below trajectory-shape signals (AURC $0.311$;
Acc@$73.7\%$ $68.8\%$).

#### Strict Yes/No self-evaluation baseline.

We also run a stricter single-token protocol in
figures/run_yesno_self_eval_baseline.py: the model must answer
“Yes” or “No” to whether its own final answer is correct.
Confidence is taken as $P(\text{Yes})$ from first-token logits when available,
with deterministic fallback to parsed Yes/No output.
At 73.7% coverage, this baseline reaches $63.3\%$ answered-set accuracy,
Spearman $\rho{=}{-}0.019$, and AURC $=0.395$ with only 2.0 verifier
tokens/problem on average (total $236.7$ including the base chain), indicating
very low cost but weak discrimination.

### A.22 Token Accounting Details and Fairness

This section reconciles the two budget comparisons used in the paper:
(1) *high-budget* reference points (SC@10/SC@40), and
(2) *near-equal-budget* empirical baselines (SC@3/SC@5/ESC).

For our method, token usage decomposes as

$$ $T_{\text{ours}}=T_{\text{base-chain}}+m\,\bar{N}\,\bar{L}_{\text{short}},$ (3) $$

where $m{=}5$ is samples per prefix, $\bar{N}$ is mean number of prefixes,
and $\bar{L}_{\text{short}}$ is mean short-completion length.
On GSM8K (Qwen run, $n{=}300$), $\bar{N}{=}4.91$ and the measured base-chain
length is $\bar{T}_{\text{base-chain}}{=}234.7$ tokens/problem.

**Table 15: Unified token accounting on GSM8K ($n{=}300$). “Measured” values come from logged generations; “configured” values are fixed decoding settings.**
| Quantity | Value | Type |
| --- | --- | --- |
| Base-chain tokens $\bar{T}_{\text{base-chain}}$ | $234.7$ | measured |
| Mean step prefixes $\bar{N}$ | $4.91$ | measured |
| Per-prefix samples $m$ | $5$ | configured |
| Short completion cap | $150$ | configured |
| Reported total $T_{\text{ours}}$ | $\approx 1500$ | reported (main text) |
| SC@3 tokens/problem | $831.4$ | measured |
| SC@5 tokens/problem | $1385.7$ | measured |
| ESC-sim tokens/problem | $673.6$ | measured |
| Self-judgment tokens/problem | $242.3$ | measured |

Interpreting these together: the “$\approx 1{,}500$” figure is the operating
budget for our selective policy, while SC/ESC costs are directly measured at
full coverage. Thus, claims against SC@10/SC@40 are high-budget references,
and claims against SC@5/ESC are the strict empirical near-equal-budget
comparisons.