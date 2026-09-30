# What does a diffusion policy's uncertainty actually measure?

A small, self-contained study of **denoising-sample dispersion** as an uncertainty
signal for language-conditioned manipulation policies.

**Headline finding.** The dispersion between denoising samples of a diffusion policy
tracks *task ambiguity* well (r = 0.81 against the conditional spread of human
demonstrations) but is **almost unresponsive to input validity**. Across three visual
degradations and one semantic one, the action error grows by a factor of 3.5 to 7 while
the dispersion stays within ±35% of its clean value, moving up under some degradations
and down under others.

Using this signal as a degradation or out-of-distribution detector — as is sometimes
suggested — is therefore not sound. It measures *aleatoric* uncertainty, not
*epistemic* uncertainty.

All experiments run on a single T4 GPU in under two hours.

---

## 1. Motivation

Diffusion policies have become the standard action head for imitation learning in
manipulation, because human demonstrations are multimodal and a regression trained
with MSE collapses to the mean of valid behaviours.

A diffusion policy also offers something a deterministic policy does not: sampling
*N* action chunks from the same observation and measuring their dispersion gives an
uncertainty estimate **for free** — no ensemble, no auxiliary head, no labels. The
appeal is obvious for real-robot deployment, where a policy that knows when it does
not know can slow down, ask for help, or trigger data collection.

This repository asks whether that signal is trustworthy. The answer is: partly, and
in a way that matters.

### 1.1 Where this question came from

I started from LIBERO (Liu et al., NeurIPS Datasets & Benchmarks 2023 -
https://papers.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf) and surveyed
the work citing it, along two lines that rarely meet. Continual-learning methods for
manipulation policies — M2Distill, CRL-VLA — measure how much of an earlier task
survives after learning a new one, but evaluate under standard benchmark conditions.
Robustness benchmarks — LIBERO-Plus, LIBERO-PRO — perturb camera views, object
layouts and instructions, but evaluate fixed policies rather than policies after each
stage of adaptation. The gap between them is *robustness-conditioned forgetting*:
whether a method retains success **under shift**, not only in the original task setup.

Testing that directly requires closed-loop rollouts after every adaptation stage,
which is a larger undertaking than this repository. What I did first was to check a
necessary precondition: if a policy is going to be evaluated under observation shift,
does it at least *notice* the shift? The answer turned out to be no, and that is what
is reported here. The continual-learning study remains the intended next step
(Section 6).

*(This gap was identified from a review of representative citing work, not from an
exhaustive citation search; novelty has not been fully verified.)*

---

## 2. Pilot study: is the signal calibrated in-distribution?

**Data.** `lerobot/aloha_sim_insertion_human` — 25,000 frames of bimanual peg
insertion from human teleoperation, 14-dimensional joint state. Forty episodes were
resampled onto a common normalised time grid of 200 points.

**Setup.** A conditional DDPM (T = 100) generates a chunk of H = 16 future values of
one joint, conditioned on the current observation and normalised time. Rollout
executes a full chunk before re-conditioning.

**Qualitative result.** With the same initial condition, MSE rollouts collapse onto a
single trajectory that matches no demonstration, while diffusion rollouts separate
into distinct branches that follow individual demonstration modes. This reproduces
the standard motivation for diffusion policies on real data.

**Calibration.** For each state along a reference trajectory, 50 chunks were sampled
and their dispersion compared against two ground-truth quantities:

| ground truth | Pearson r |
|---|---|
| marginal spread of demonstrations at time *t* | 0.21 |
| **conditional spread** — dispersion of the next H steps among the 25 nearest real demonstrations in (t, y) | **0.81** |

The first comparison is the wrong one, and it is reported deliberately. A policy
conditioned on the current observation should not be as uncertain as the marginal
distribution of all demonstrations; knowing where the arm *is* already identifies the
branch. Correcting the ground truth from marginal to conditional dispersion moved the
correlation from 0.21 to 0.81 without touching the model.

**Conclusion of the pilot.** In-distribution, the signal is well calibrated: it peaks
where the demonstrations genuinely diverge and collapses once the policy has committed
to a branch.

---

## 3. Main study: a minimal VLA on LIBERO-Goal

### 3.1 Why LIBERO-Goal

`lerobot/libero_goal_image` — 428 episodes, 52,042 frames, 10 tasks; two RGB views at
256×256 (external and wrist), 8-dimensional state, 7-dimensional action.

LIBERO-Goal is the suite in which **objects and layout are identical across tasks and
only the goal changes**. Four of its ten instructions begin "put the bowl…" and end in
different places. In the opening frames the scene is visually near-identical and the
instruction is the only signal that disambiguates the goal. This makes it possible to
test whether language has actually entered the policy, rather than being a decorative
input the model can ignore.

Training used the first 20,000 streamed frames (163 episodes). All ten tasks are
present and reasonably balanced:

```
task:   0     1     2     3     4     5     6     7     8     9
n:   1806  2592  3299  1645  2481  1804  1706  1193  1374  2100
```

### 3.2 Architecture

```
 RGB (external view) ──► ResNet18, frozen      ──► 512 ┐
 instruction         ──► MiniLM-L6-v2, frozen  ──► 384 ├──► conditioning (904)
 proprioception      ─────────────────────────►    8 ┘         │
                                                                ▼
                                             ┌──────────────────────────────┐
                                             │  ε-network (MLP)             │
                                             │  DDPM, T = 100               │  ← only trained part
                                             └──────────────┬───────────────┘
                                                            ▼
                                              action chunk (16 × 7 = 112)
```

Both encoders are frozen and their outputs precomputed once, which reduces training to
a conditional DDPM over a 112-dimensional target. This is a deliberate scoping choice,
not a fallback: it makes the whole study reproducible on free-tier hardware, and the
question under investigation concerns the action head, not representation learning.

Training: 17,393 (condition, chunk) pairs, 20,000 steps, AdamW at 3e-4, batch 256.
Loss 0.99 → 0.16.

### 3.3 Does language control the action?

Same image, same proprioceptive state, different instruction. The metric is a ratio:

```
        mean pairwise distance between per-instruction mean trajectories
ratio = ---------------------------------------------------------------
                  mean within-instruction sample dispersion
```

A ratio above 1 means changing the sentence moves the trajectory more than sampling
noise does.

| condition | ratio |
|---|---|
| **different instructions** (5 tasks, 20 states, n = 32) | **2.70 ± 0.30** |
| paraphrases of "put the bowl on the plate" | 1.14 ± 0.16 |
| paraphrases of "turn on the stove" | 1.79 ± 0.14 |
| paraphrases of "open the middle drawer of the cabinet" | 0.68 ± 0.13 |

Paraphrases are the negative control: three phrasings of the same goal
("put the bowl on the plate" / "place the bowl onto the plate" / "set the bowl down on
the plate"). Genuinely different goals move the trajectory 2.2× more than paraphrases
do (2.70 against a paraphrase mean of 1.20), so **the policy responds mainly to meaning
rather than to surface form**.

The control is imperfect, and the imperfection is worth naming. The stove paraphrases
("switch on the stove", "start the stove") reach 1.79, which is above sampling noise:
MiniLM does not place them as close together as it places the bowl paraphrases, so part
of that ratio is encoder geometry rather than policy behaviour. The drawer task sits
lowest (0.68), plausibly because the action is kinematically constrained — any sensible
phrasing produces the same motion. A cleaner version of this experiment would select
paraphrases by their embedding distance rather than by intuition.

### 3.4 Does the policy notice when its inputs degrade?

Evaluation set: 2,000 streamed frames (tasks 0–7 present), 400 sampled states, 8
denoising samples per state. Each modality is degraded in isolation; MAE is measured
against the ground-truth action chunk.

| degradation | level | MAE | dispersion |
|---|---|---|---|
| Gaussian blur | 0 → 16 | 0.068 → 0.480 (**×7.0**) | 0.0579 → 0.0507 (×0.88) |
| central occlusion | 0 → 0.8 | 0.068 → 0.285 (**×4.2**) | 0.0575 → 0.0379 (×0.66) |
| darkening | 0 → 0.9 | 0.068 → 0.317 (**×4.7**) | 0.0576 → 0.0674 (×1.17) |

The quantity to read here is the **ratio of relative changes**, not a correlation. Error
multiplies by between 4.2 and 7.0; dispersion stays inside a factor of 0.66 to 1.17 of
its clean value, and does not even agree on a direction across the three degradations.

**Why not report a correlation.** An earlier training run of exactly this pipeline gave
corr(MAE, dispersion) = −0.31 (blur), −0.34 (occlusion), −0.93 (darkness); the run
reported above gives −0.44, −0.95 and **+0.98**. The darkness correlation changes sign
between two runs that differ only in initialisation. With five degradation levels per
condition, that correlation is not an estimable quantity, and reporting it would invite
a reader to reproduce the opposite number. The magnitude ratios, by contrast, are
consistent across both runs: dispersion never moves more than ~35% while error moves by
several hundred percent.

**Within-run precision.** The dispersion estimate itself is precise; it is the training
run that varies. Repeating the blur measurement with five sampling seeds on a fixed
checkpoint gives 0.0576 ± 0.0003 at level 0 and 0.0506 ± 0.0000 at level 16, so the
small movements in the table are not sampling noise — they are simply small.

**Cross-modality control.** Replacing each state's instruction with the next task's
instruction — a semantically wrong but perfectly well-formed sentence — raises MAE from
0.069 to 0.239 (×3.5) while dispersion moves from 0.0576 to 0.0541 (×0.94). The effect
is not an artefact of the visual channel.

---

## 4. Interpretation

Denoising-sample dispersion measures **how many valid ways there are to continue**, not
**whether the inputs make sense**.

In-distribution the two coincide often enough to look like a general uncertainty
estimate: ambiguous states are exactly the states where demonstrations disagree, so the
signal is well calibrated (r = 0.81).

Out-of-distribution they come apart. A blurred image or a wrong instruction does not
produce an ambiguous conditioning vector; it produces a conditioning vector in a region
the ε-network never saw, where it has learned an arbitrary but *decisive* mapping. The
model is not uncertain — it is confidently elsewhere.

The magnitudes make the practical point sharper than any sign would. A gate that
triggered on dispersion would need to resolve a 17% change in order to catch an error
that has already grown by 366%. Even in the one condition where the signal moves in the
right direction, it moves far too little to be actionable.

This is the aleatoric/epistemic distinction, made concrete on a robot policy: the
signal captures the first and is blind to the second. For deployment, the second is the
one that matters, because a fouled sensor or a misheard command is precisely the
situation in which silent failure is most expensive.

---

## 5. Limitations

These bound what the results support, and are stated in full deliberately.

1. **No closed-loop evaluation.** All numbers are action-prediction error on held-out
   frames, not task success rate. Success would require rollouts in robosuite/MuJoCo.
   Prediction error is a proxy and may not order methods the same way success does.
2. **Evaluation frames overlap with training frames.** The degradation study reuses
   frames seen during training. This is acceptable for measuring the *relative* change
   in error as degradation increases, but it is not a generalisation measurement.
3. **Frozen encoders.** Whether the same blindness holds for an end-to-end VLA, where
   the visual encoder is itself adapted, is untested and is the most important open
   question here.
4. **One task suite; one training run per reported configuration.** Sampling seeds are
   averaged where error bars are given, but the training run itself is a single draw.
   Section 3.4 shows why this matters: the sign of corr(MAE, dispersion) is not stable
   across runs even when the magnitude ratios are.
5. **Fixed horizon, no receding-horizon execution.** Real diffusion policies re-plan
   before the chunk is exhausted; chunk boundaries are visible in rollouts.
6. **Partial task coverage in evaluation** — tasks 8 and 9 are absent from the
   evaluation split.
7. **Novelty is not fully established.** The gap described in Section 1.1 was
   identified from representative citing work, not from an exhaustive citation
   search. A targeted search would be the next step before claiming priority.
8. **A scaling attempt failed.** Extending the ALOHA sampler from one joint to all
   fourteen diverged numerically (dispersion ~1e10), most likely requiring clamping of
   the predicted x₀ in the reverse step and a lower learning rate. This is unresolved
   and reported rather than omitted.

---

## 6. What would come next

- **End-to-end encoders**, to test whether adaptation of the visual backbone changes
  the picture (limitation 3).
- **A signal that does detect input degradation** — reconstruction error in the encoder
  latent space, or a density estimate over the conditioning vector — evaluated with the
  same protocol, so the two can be compared on equal terms.
- **Closed-loop validation**: does a policy gated on an epistemic signal actually avoid
  failures, and at what cost in interventions?
- **Continual adaptation under shift.** If a policy fails silently under degradation,
  then continual-learning methods evaluated only under nominal conditions may be
  reporting retention that does not survive realistic observation shift. Evaluating old
  tasks after every adaptation stage, both nominally and under held-out shifts, is the
  natural extension of this work.

---

## 7. Reproducing

```
libero_vla_uncertainty.ipynb   # the LIBERO-Goal study (Sections 3.1-3.4), end to end
```

The notebook runs top to bottom on a Kaggle T4: it streams the dataset, precomputes
the frozen-encoder features, trains the ε-network, then reproduces the
language-control and degradation experiments. Checkpoints (`feats.pt`, `vla.pt`) are
written to the working directory so the later sections can be re-run without
retraining.

The ALOHA pilot of Section 2 was run separately and is not included here; it uses the
same DDPM code path on `lerobot/aloha_sim_insertion_human` with a single joint as the
target.

Dependencies: `torch`, `datasets`, `transformers`, `torchvision`, `scipy`,
`matplotlib`. No robot simulator required.

---

*Giulia Avanzato — MSc Computer Engineering (Cybersecurity and Artificial
Intelligence), University of Cagliari.*
