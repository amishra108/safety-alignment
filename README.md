# Safety Alignment in Audio-Language Models

This repository contains research on safety alignment in audio-language models, developed as part of a fourth-year research project at the University of Cambridge.

The project investigates whether audio-language models recognise harmful instructions consistently across text and speech, how harmfulness is represented internally, and whether these representations can be used to influence model safety behaviour through activation steering.

## Models

The primary models investigated are:

- Qwen2-Audio-7B-Instruct
- Qwen2.5-Omni-7B
- Kimi-Audio-7B-Instruct
- Audio Flamingo 3

Additional models are used for speech synthesis, transcription, response adjudication, and other supporting tasks.

## Safety Benchmarks

The project evaluates models using several existing benchmarks:

- AdvBench
- XSTest
- Jailbreak-AudioBench
- SACRED-Bench
- ALMGuard

Evaluations consider harmful compliance, refusal behaviour, and inappropriate refusal of benign requests.

## Multilingual Audio Jailbreak Clean Benchmark

A major component of the project is the construction of a multilingual benchmark for controlled text-versus-audio safety evaluation.

The benchmark contains 100 harmful and 100 benign underlying English requests, expanded across four languages:

- English
- Chinese
- Russian
- Spanish

Each request appears under four prompting conditions:

- Original
- Literal
- Narrative framing
- Content dilution

Both text and synthesised audio versions are prepared, producing 6,400 evaluation inputs.

The benchmark enables comparisons across models, languages, prompting conditions, and modalities while preserving the underlying source request.

The preparation pipeline includes translation, speech synthesis, manifest construction, model evaluation, and response analysis.

## Harmfulness Directions in Representation Space

The project investigates whether harmful and benign prompts are distinguishable in the hidden representations of audio-language models.

Representations are extracted from multiple model layers and analysed using:

- Paired harmful-minus-benign representation differences.
- Singular value decomposition (SVD).
- Projection analysis.
- Cosine similarity.
- Linear probing.
- Prompt-disjoint cross-validation.

A central question is whether harmfulness-related directions identified using text prompts are similar to those identified using matched audio prompts.

The experiments also examine how representations change between modalities, whether these changes align with harmfulness-related directions, and how the relevant information develops across model layers.

## Activation Steering

Candidate harmfulness directions are subsequently used for activation steering.

At selected layers, hidden representations are modified during inference by adding a scaled steering vector:

\[
h' = h + \alpha v
\]

where \(v\) is a selected direction and \(\alpha\) controls the intervention strength.

The experiments compare original and steered responses to investigate changes in harmful compliance, refusal behaviour, and benign over-refusal.

These experiments examine whether the directions identified through representation analysis can causally influence model behaviour, rather than merely correlate with harmfulness labels.

## Repository Structure

- `scripts/preprocessing/` — Dataset preparation, translation, and audio synthesis.
- `scripts/initial_benchmark/` — Benchmark evaluation and response analysis.
- `scripts/exploring_safety/` — Representation extraction, linear probing, SVD, and cross-modal analysis.
- `scripts/wk3/` — Harmfulness-direction and activation steering experiments.
- `scripts/wk3_archive/` — Earlier experimental scripts.
- `data/` — Selected dataset manifests, metadata, and descriptions.
- `models/` — Model descriptions and intended uses.
- `outputs/` — Selected experimental outputs.
- `results/` — Evaluation results and analysis artefacts.

Third-party literature code is not redistributed. Where applicable, literature directories contain explanatory placeholder files.

Large model weights, raw audio collections, and other storage-intensive resources are excluded.

## Using the Codebase

The repository consists of research scripts rather than a standalone Python package.

To reproduce an experiment, obtain the required pretrained models and datasets, prepare the benchmark inputs using `scripts/preprocessing/`, and run the relevant evaluation or analysis scripts.

The representation analysis and activation steering experiments are primarily located in `scripts/exploring_safety/` and `scripts/wk3/`.

The scripts were developed for GPU-enabled research environments and may require adjustments to local model paths, dependencies, and hardware configurations.

## Research Scope

The project focuses on understanding safety alignment across audio and text modalities, identifying harmfulness-related internal representations, and investigating their potential role in controlling model behaviour.

Representational separability does not necessarily establish a unique internal concept of harmfulness, and steering effects must be evaluated against both harmful and benign behaviour.
