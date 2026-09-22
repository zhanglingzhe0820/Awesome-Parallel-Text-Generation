# Awesome-Parallel-Text-Generation

## Our Survey

[A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](https://arxiv.org/abs/2508.08712) · [PDF](https://arxiv.org/pdf/2508.08712)

A systematic survey of token-level and task-level parallel text generation. This index follows the revised manuscript's taxonomy; the linked arXiv version may precede the manuscript revision.

## Taxonomy

- [Token-Level Parallelism](#token-level-parallelism)
  - [AR-Based](#ar-based)
    - [Drafter Design](#drafter-design)
    - [Verification Strategies](#verification-strategies)
  - [Non-AR-Based](#non-ar-based)
    - [One-Shot Generation](#one-shot-generation)
    - [Masked Generation](#masked-generation)
    - [Edit-Based Refinement](#edit-based-refinement)
- [Task-Level Parallelism](#task-level-parallelism)
  - [Content Decomposition](#content-decomposition)
    - [Query Decomposition](#query-decomposition)
    - [Structure Planning](#structure-planning)
  - [Process Orchestration](#process-orchestration)
    - [Plan Execution](#plan-execution)
    - [Branch Reasoning](#branch-reasoning)
- [Complementary Acceleration](#complementary-acceleration)
- [Related Surveys](#related-surveys)

Token-level and task-level parallelism are composable: concurrent tasks can each use AR-based, Non-AR-based, or hybrid decoding. Categories organize mechanisms, not mutually exclusive complete systems. Drafting and verification can also be combined, and masked generation overlaps with masked-edit refinement. Papers are indexed by the mechanism discussed in the survey. A dash in the Code column means that no code link has been verified for that entry.

## Token-Level Parallelism

### AR-Based

Draft-and-verify accelerates a target autoregressive decoder through multi-token proposals and parallel verification. Greedy verification and target-distribution-preserving speculative sampling have different correctness guarantees.

#### Drafter Design

##### External Drafter

| Paper | Venue | Code |
| --- | :---: | :---: |
| [Speculative Decoding with Big Little Decoder](https://arxiv.org/abs/2302.07863)  |  NeurIPS 2023  | [![Github](https://img.shields.io/github/stars/kssteven418/BigLittleDecoder?style=flat)](https://github.com/kssteven418/BigLittleDecoder) |
| [Cascade speculative drafting for even faster llm inference](https://arxiv.org/abs/2312.11462)  |  NeurIPS 2023  | [![Github](https://img.shields.io/github/stars/lfsszd/CS-Drafting?style=flat)](https://github.com/lfsszd/CS-Drafting) |
| [Distillspec: Improving speculative decoding via knowledge distillation](https://arxiv.org/abs/2310.08461)  |  ICLR 2024  | - |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)  |  ICML 2023  | [![Github](https://img.shields.io/github/stars/romsto/Speculative-Decoding?style=flat)](https://github.com/romsto/Speculative-Decoding) |
| [Online Speculative Decoding](https://arxiv.org/abs/2310.07177)  |  ICML 2024  | [![Github](https://img.shields.io/github/stars/LiuXiaoxuanPKU/OSD?style=flat)](https://github.com/LiuXiaoxuanPKU/OSD) |
| [Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation](https://arxiv.org/abs/2307.02150)  |  EMNLP 2023  | [![Github](https://img.shields.io/github/stars/hemingkx/SpecDec?style=flat)](https://github.com/hemingkx/SpecDec) |
| [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)  |  arxiv 2023  | [![Github](https://img.shields.io/github/stars/shreyansh26/Speculative-Sampling?style=flat)](https://github.com/shreyansh26/Speculative-Sampling) |
| [Accelerating LLM Inference with Staged Speculative Decoding](https://arxiv.org/abs/2308.04623) | arXiv 2023 | — |

##### Self-Drafting via Partial Computation

BranchyNet is an early-exit precursor, not itself an LLM self-speculative decoding system.

| Paper | Venue | Code |
| --- | :---: | :---: |
| [Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168)  |  ACL 2024  | [![Github](https://img.shields.io/github/stars/dilab-zju/self-speculative-decoding?style=flat)](https://github.com/dilab-zju/self-speculative-decoding) |
| [Speculative Decoding via Early-Exiting for Faster LLM Inference with Thompson Sampling Control Mechanism](https://arxiv.org/abs/2406.03853)  |  ACL 2024  | - |
| [Kangaroo: Lossless self-speculative decoding via double early exiting](https://openreview.net/forum?id=lT3oc04mDp)  |  NeurIPS 2024  | [![Github](https://img.shields.io/github/stars/Equationliu/Kangaroo?style=flat)](https://github.com/Equationliu/Kangaroo) |
| [Layer-skip: Enabling early-exit inference and self-speculative decoding](https://arxiv.org/abs/2404.16710)  |  ACL 2024  | [![Github](https://img.shields.io/github/stars/facebookresearch/LayerSkip?style=flat)](https://github.com/facebookresearch/LayerSkip) |
| [Speed: speculative pipelined execution for efficient decoding](https://arxiv.org/abs/2310.12072)  |  NeurIPS-ENLSP 2023  | - |
| [Swift: On-the-fly self-speculative decoding for llm inference acceleration](https://arxiv.org/abs/2410.06916)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/hemingkx/SWIFT?style=flat)](https://github.com/hemingkx/SWIFT) |
| [CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](https://arxiv.org/abs/2510.26843)  |  NeurIPS 2025  | - |
| [BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks](https://arxiv.org/abs/1709.01686) | ICPR 2016 | — |

##### Self-Drafting via Parallel Prediction

Multi-token prediction is part of drafter design, rather than a separate category alongside draft-and-verify.

| Paper | Venue | Code |
| --- | :---: | :---: |
| [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)  |  ICML 2024  | [![Github](https://img.shields.io/github/stars/SafeAILab/EAGLE?style=flat)](https://github.com/SafeAILab/EAGLE) |
| [Falcon: Faster and Parallel Inference of Large Language Models through Enhanced Semi-Autoregressive Drafting and Custom-designed Decoding Tree](https://ojs.aaai.org/index.php/AAAI/article/view/34566)  |  AAAI 2025  | [![Github](https://img.shields.io/github/stars/Bestpay-inc/Falcon?style=flat)](https://github.com/Bestpay-inc/Falcon) |
| [Learning Harmonized Representations for Speculative Sampling](https://arxiv.org/abs/2408.15766)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/HArmonizedSS/HASS?style=flat)](https://github.com/HArmonizedSS/HASS) |
| [Hydra: Sequentially-dependent draft heads for medusa decoding](https://arxiv.org/abs/2402.05109)  |  COLM 2024  | [![Github](https://img.shields.io/github/stars/zankner/Hydra?style=flat)](https://github.com/zankner/Hydra) |
| [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)  |  ICML 2024  | [![Github](https://img.shields.io/github/stars/FasterDecoding/Medusa?style=flat)](https://github.com/FasterDecoding/Medusa) |
| [Mixture of Attentions for Speculative Decoding](https://arxiv.org/abs/2410.03804)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/huawei-noah/HEBO?style=flat)](https://github.com/huawei-noah/HEBO) |
| [Pass: Parallel speculative sampling](https://arxiv.org/abs/2311.13581)  |  NeurIPS-ENLSP 2023  | - |
| [L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](https://arxiv.org/abs/2505.17505)  |  arxiv 2025  | - |
| [On multi-token prediction for efficient LLM inference](https://arxiv.org/abs/2502.09419)  |  arxiv 2025  | - |
| [Multi-Token Prediction Needs Registers](https://arxiv.org/abs/2505.10518)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/nasosger/MuToR?style=flat)](https://github.com/nasosger/MuToR) |
| [Blockwise Parallel Decoding for Deep Autoregressive Models](https://proceedings.neurips.cc/paper/2018/hash/c4127b9194fe8562c64dc0f5bf2c93bc-Abstract.html)  |  NeurIPS 2018  | - |
| [Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential](https://arxiv.org/abs/2507.11851)  |  arxiv 2025  | - |
| [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063)  |  EMNLP 2020  | [![Github](https://img.shields.io/github/stars/microsoft/ProphetNet?style=flat)](https://github.com/microsoft/ProphetNet) |
| [Better & faster large language models via multi-token prediction](https://arxiv.org/abs/2404.19737)  |  ICML 2024  | - |
| [Deepseek-v3 technical report](https://arxiv.org/abs/2412.19437)  |  arxiv 2024  | [![Github](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-V3?style=flat)](https://github.com/deepseek-ai/DeepSeek-V3) |
| [MiMo: Unlocking the Reasoning Potential of Language Model--From Pretraining to Posttraining](https://arxiv.org/abs/2505.07608)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/xiaomimimo/MiMo?style=flat)](https://github.com/xiaomimimo/MiMo) |
| [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) | NeurIPS 2025 | [![Github](https://img.shields.io/github/stars/SafeAILab/EAGLE?style=flat)](https://github.com/SafeAILab/EAGLE) |

Drafting controls and complementary speculative extensions include adaptive draft lengths, alignment, diffusion proposals, and reasoning-level speculation; they are not additional mutually exclusive drafter types.

| Paper | Venue | Code |
| --- | :---: | :---: |
| [A Drop-in Solution for On-the-fly Adaptation of Speculative Decoding in Large Language Models](https://aclanthology.org/2025.acl-long.482)  |  ACL 2025  | - |
| [Specdec++: Boosting speculative decoding via adaptive candidate lengths](https://arxiv.org/abs/2405.19715)  |  COLM 2025  | [![Github](https://img.shields.io/github/stars/Kaffaljidhmah2/SpecDec_pp?style=flat)](https://github.com/Kaffaljidhmah2/SpecDec_pp) |
| [SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning](https://arxiv.org/abs/2504.07891)  |  NeurIPS 2025  | [![Github](https://img.shields.io/github/stars/ruipeterpan/specreason?style=flat)](https://github.com/ruipeterpan/specreason) |
| [Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs](https://arxiv.org/abs/2512.20573)  |  arXiv 2025  | [![Github](https://img.shields.io/github/stars/ruipeterpan/failfast?style=flat)](https://github.com/ruipeterpan/failfast) |
| [Scaling Speculative Decoding with Lookahead Reasoning](https://arxiv.org/abs/2506.19830)  |  NeurIPS 2025  | [![Github](https://img.shields.io/github/stars/hao-ai-lab/LookaheadReasoning?style=flat)](https://github.com/hao-ai-lab/LookaheadReasoning) |
| [Griffin: Effective token alignment for faster speculative decoding](https://arxiv.org/abs/2502.11018)  |  NeurIPS 2025  | [![Github](https://img.shields.io/github/stars/hsj576/GRIFFIN?style=flat)](https://github.com/hsj576/GRIFFIN) |
| [Speculative diffusion decoding: Accelerating language generation through diffusion](https://aclanthology.org/2025.naacl-long.601/) | NAACL 2025 | — |

#### Verification Strategies

##### Efficient Verification

Tree/graph verification and scheduling methods reduce verification or end-to-end draft-and-verify latency.

| Paper | Venue | Code |
| --- | :---: | :---: |
| [Adaptive Draft-Verification for Efficient Large Language Model Decoding](https://ojs.aaai.org/index.php/AAAI/article/view/34647)  |  AAAI 2025  | [![Github](https://img.shields.io/github/stars/liuxukun2000/Adaptix?style=flat)](https://github.com/liuxukun2000/Adaptix) |
| [Dynamic Depth Decoding: Faster Speculative Decoding for LLMs](https://arxiv.org/abs/2409.00142)  |  arxiv 2024  | - |
| [DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure](https://arxiv.org/abs/2410.11744)  |  WWW 2025  | - |
| [Eagle-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)  |  EMNLP 2024  | [![Github](https://img.shields.io/github/stars/SafeAILab/EAGLE?style=flat)](https://github.com/SafeAILab/EAGLE) |
| [Graph-Structured Speculative Decoding](https://arxiv.org/abs/2407.16207)  |  ACL 2024  | [![Github](https://img.shields.io/github/stars/gzhch/gsd?style=flat)](https://github.com/gzhch/gsd) |
| [OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure](https://arxiv.org/abs/2406.17276)  |  TACL 2025  | [![Github](https://img.shields.io/github/stars/Jikai0Wang/OPT-Tree?style=flat)](https://github.com/Jikai0Wang/OPT-Tree) |
| [Ouroboros: Generating Longer Drafts Phrase by Phrase for Faster Speculative Decoding](https://aclanthology.org/2024.emnlp-main.742/)  | EMNLP 2024 | [![Github](https://img.shields.io/github/stars/thunlp/Ouroboros?style=flat)](https://github.com/thunlp/Ouroboros) |
| [Parallel Speculative Decoding with Adaptive Draft Length](https://arxiv.org/abs/2408.11850)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/smart-lty/ParallelSpeculativeDecoding?style=flat)](https://github.com/smart-lty/ParallelSpeculativeDecoding) |
| [PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation](https://arxiv.org/abs/2407.11798)  |  SC 2024  | [![Github](https://img.shields.io/github/stars/AutonomicPerfectionist/PipeInfer?style=flat)](https://github.com/AutonomicPerfectionist/PipeInfer) |
| [Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding](https://arxiv.org/abs/2307.05908)  |  TMLR 2024  | - |
| [Sequoia: Scalable, robust, and hardware-aware speculative decoding](https://arxiv.org/abs/2402.12374)  |  arxiv 2024  | [![Github](https://img.shields.io/github/stars/Infini-AI-Lab/Sequoia?style=flat)](https://github.com/Infini-AI-Lab/Sequoia) |
| [Generation meets verification: Accelerating large language model inference with smart parallel auto-correct decoding](https://arxiv.org/abs/2402.11809)  |  ACL 2024  | [![Github](https://img.shields.io/github/stars/cteant/SPACE?style=flat)](https://github.com/cteant/SPACE) |
| [Specinfer: Accelerating generative large language model serving with tree-based speculative inference and verification](https://arxiv.org/abs/2305.09781)  |  ASPLOS 2024  | [![Github](https://img.shields.io/github/stars/flexflow/FlexFlow?style=flat)](https://github.com/flexflow/FlexFlow) |
| [STree: Speculative Tree Decoding for Hybrid State-Space Models](https://arxiv.org/abs/2505.14969)  |  NeurIPS 2025  | [![Github](https://img.shields.io/github/stars/wyc1997/stree?style=flat)](https://github.com/wyc1997/stree) |

##### Accurate Verification

These methods improve acceptance or verification decisions. Relaxed or learned acceptance rules do not automatically preserve the target sampling distribution.

| Paper | Venue | Code |
| --- | :---: | :---: |
| [Block Verification Accelerates Speculative Decoding](https://arxiv.org/abs/2403.10444)  |  ICLR 2025  | - |
| [Dynamic-Width Speculative Beam Decoding for Efficient LLM Inference](https://arxiv.org/abs/2409.16560)  |  AAAI 2025  | [![Github](https://img.shields.io/github/stars/ZongyueQin/DSBD?style=flat)](https://github.com/ZongyueQin/DSBD) |
| [Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment](https://arxiv.org/abs/2501.19309)  |  ICLR 2025  | - |
| [Optimized multi-token joint decoding with auxiliary model for llm inference](https://arxiv.org/abs/2407.09722)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/ZongyueQin/MTAD?style=flat)](https://github.com/ZongyueQin/MTAD) |
| [ProPD: Dynamic Token Tree Pruning and Generation for LLM Parallel Decoding](https://arxiv.org/abs/2402.13485)  |  ICCAD 2024  | - |
| [REST: Retrieval-based speculative decoding](https://arxiv.org/abs/2311.08252)  |  NAACL 2024  | [![Github](https://img.shields.io/github/stars/FasterDecoding/REST?style=flat)](https://github.com/FasterDecoding/REST) |
| [Recursive Speculative Decoding: Accelerating LLM Inference via Sampling without Replacement](https://arxiv.org/abs/2402.14160)  |  ICLR-LLMA 2024  | - |
| [SpecTr: Fast Speculative Decoding via Optimal Transport](https://arxiv.org/abs/2310.15141)  |  NeurIPS 2023  | - |
| [Towards Optimal Multi-draft Speculative Decoding](https://arxiv.org/abs/2502.18779) | ICLR 2025 | — |
| [Multi-Draft Speculative Sampling: Canonical Decomposition and Theoretical Limits](https://arxiv.org/abs/2410.18234) | arXiv 2024 | — |
| [A Theoretical Perspective for Speculative Decoding Algorithm](https://arxiv.org/abs/2411.00841) | NeurIPS 2024 | — |

### Non-AR-Based

#### One-Shot Generation

| Paper | Venue | Code |
| --- | :---: | :---: |
| [Non-autoregressive neural machine translation](https://arxiv.org/abs/1711.02281)  |  ICLR 2018  | [![Github](https://img.shields.io/github/stars/salesforce/nonauto-nmt?style=flat)](https://github.com/salesforce/nonauto-nmt) |
| [End-to-end non-autoregressive neural machine translation with connectionist temporal classification](https://arxiv.org/abs/1811.04719)  |  EMNLP 2018  |  |
| [Lava nat: A non-autoregressive translation model with look-around decoding and vocabulary attention](https://arxiv.org/abs/2002.03084)  |  arxiv 2025  | - |
| [AligNART: Non-autoregressive neural machine translation by jointly learning to estimate alignment and translate](https://arxiv.org/abs/2109.06481)  |  EMNLP 2021  | - |
| [Guiding non-autoregressive neural machine translation decoding with reordering information](https://ojs.aaai.org/index.php/AAAI/article/view/17618)  |  AAAI 2021  | [![Github](https://img.shields.io/github/stars/ranqiu92/ReorderNAT?style=flat)](https://github.com/ranqiu92/ReorderNAT) |
| [Non-monotonic latent alignments for ctc-based non-autoregressive machine translation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/35f805e65c77652efa731edc10c8e3a6-Abstract-Conference.html)  |  NeurIPS 2022  | [![Github](https://img.shields.io/github/stars/ictnlp/NMLA-NAT?style=flat)](https://github.com/ictnlp/NMLA-NAT) |
| [DePA: Improving Non-autoregressive Translation with Dependency-Aware Decoder](https://aclanthology.org/2023.iwslt-1.47/)  | IWSLT 2023 | [![Github](https://img.shields.io/github/stars/zhanjiaao/NAT_DePA?style=flat)](https://github.com/zhanjiaao/NAT_DePA) |
| [Directed acyclic transformer for non-autoregressive machine translation](https://proceedings.mlr.press/v162/huang22m)  |  ICML 2022  | [![Github](https://img.shields.io/github/stars/thu-coai/DA-Transformer?style=flat)](https://github.com/thu-coai/DA-Transformer) |
| [Viterbi decoding of directed acyclic transformer for non-autoregressive machine translation](https://arxiv.org/abs/2210.05193)  |  EMNLP 2022  | [![Github](https://img.shields.io/github/stars/thu-coai/DA-Transformer?style=flat)](https://github.com/thu-coai/DA-Transformer) |
| [Fully Non-autoregressive Neural Machine Translation: Tricks of the Trade](https://arxiv.org/abs/2012.15833)  |  ACL-IJCNLP 2021  | - |
| [Aligned cross entropy for non-autoregressive machine translation](https://proceedings.mlr.press/v119/ghazvininejad20a.html)  |  ICML 2020  | [![Github](https://img.shields.io/github/stars/m3yrin/aligned-cross-entropy?style=flat)](https://github.com/m3yrin/aligned-cross-entropy) |
| [ngram-OAXE: Phrase-based order-agnostic cross entropy for non-autoregressive machine translation](https://arxiv.org/abs/2210.03999)  |  COLING 2022  | [![Github](https://img.shields.io/github/stars/ltencent-ailab/machine-translation?style=flat)](https://github.com/tencent-ailab/machine-translation/COLING22_ngram-OAXE) |
| [Multi-granularity optimization for non-autoregressive translation](https://arxiv.org/abs/2210.11017)  |  EMNLP 2022  | [![Github](https://img.shields.io/github/stars/yafuly/MGMO-NAT?style=flat)](https://github.com/yafuly/MGMO-NAT) |
| [Multilingual Non-Autoregressive Machine Translation without Knowledge Distillation](https://aclanthology.org/2023.findings-ijcnlp.14.pdf)  |  IJCNLP-AACL 2023  | - |
| [Tree-Structured Non-Autoregressive Decoding for Sequence-to-Sequence Text Generation](https://aclanthology.org/anthology-files/anthology-files/pdf/findings/2025.findings-emnlp.327.pdf)  |  EMNLP 2025  | [![Github](https://img.shields.io/github/stars/jipy0222/TNAD?style=flat)](https://github.com/jipy0222/TNAD) |
| [Enriching Non-Autoregressive Transformer with Syntactic and Semantic Structures for Neural Machine Translation](https://arxiv.org/abs/2101.08942) | arXiv 2021 | — |
| [Non-autoregressive text generation with pre-trained language models](https://aclanthology.org/2021.eacl-main.18/) | EACL 2021 | — |
| [One Reference Is Not Enough: Diverse Distillation with Reference Selection for Non-Autoregressive Translation](https://arxiv.org/abs/2205.14333) | NAACL 2022 | [![Github](https://img.shields.io/github/stars/ictnlp/DDRS-NAT?style=flat)](https://github.com/ictnlp/DDRS-NAT) |
| [What have we achieved on non-autoregressive translation?](https://aclanthology.org/2024.findings-acl.452/) | Findings ACL 2024 | — |
| [Non-Autoregressive Machine Translation with Latent Alignments](https://aclanthology.org/2020.emnlp-main.83/) | EMNLP 2020 | — |
| [Glancing transformer for non-autoregressive neural machine translation](https://aclanthology.org/2021.acl-long.155/) | ACL-IJCNLP 2021 | [![Github](https://img.shields.io/github/stars/FLC777/GLAT?style=flat)](https://github.com/FLC777/GLAT) |

#### Masked Generation

This category includes masked-token generation, diffusion formulations, and their training and inference optimizations, as well as related theoretical and evaluation studies. Diffusion-LM and Plaid are included as continuous-diffusion precursors, not as masked diffusion models.

| Paper | Venue | Code |
| --- | :---: | :---: |
| [Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling](https://arxiv.org/abs/2409.02908)  |  ICLR 2025  | - |
| [A continuous time framework for discrete denoising models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html)  |  NeurIPS 2022  | [![Github](https://img.shields.io/github/stars/andrew-cr/tauLDR?style=flat)](https://github.com/andrew-cr/tauLDR) |
| [Discrete diffusion modeling by estimating the ratios of the data distribution](https://arxiv.org/abs/2310.16834)  |  ICML 2024  | [![Github](https://img.shields.io/github/stars/louaaron/Score-Entropy-Discrete-Diffusion?style=flat)](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) |
| [Simplified and generalized masked diffusion for discrete data](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html) | NeurIPS 2024 | [google-deepmind/md4](https://github.com/google-deepmind/md4) · [darioShar/pytorch-md4](https://github.com/darioShar/pytorch-md4) |
| [Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference](https://arxiv.org/abs/2508.02193)  |  arxiv 2025  | - |
| [Target concrete score matching: A holistic framework for discrete diffusion](https://arxiv.org/abs/2504.16431)  |  ICML 2025  | - |
| [Score-based continuous-time discrete diffusion models](https://arxiv.org/abs/2211.16750)  |  ICLR 2023  | - |
| [Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding](https://arxiv.org/abs/2505.22618)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/NVlabs/Fast-dLLM?style=flat)](https://github.com/NVlabs/Fast-dLLM) |
| [Large language diffusion models](https://arxiv.org/abs/2502.09992)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/ML-GSAI/LLaDA?style=flat)](https://github.com/ML-GSAI/LLaDA) |
| [Beyond autoregression: Discrete diffusion for complex reasoning and planning](https://arxiv.org/abs/2410.14157)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/HKUNLP/diffusion-vs-ar?style=flat)](https://github.com/HKUNLP/diffusion-vs-ar) |
| [A reparameterized discrete diffusion model for text generation](https://arxiv.org/abs/2302.05737)  |  COLM 2024  | [![Github](https://img.shields.io/github/stars/hkunlp/reparam-discrete-diffusion?style=flat)](https://github.com/hkunlp/reparam-discrete-diffusion) |
| [Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions](https://arxiv.org/abs/2502.06768)  |  ICML 2025  | - |
| [Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking](https://arxiv.org/abs/2505.24857)  |  arxiv 2025  | - |
| [Accelerating Diffusion Large Language Models with SlowFast: The Three Golden Principles](https://arxiv.org/abs/2506.10848)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/LiangrunFlora/Slow-Fast-Sampling?style=flat)](https://github.com/LiangrunFlora/Slow-Fast-Sampling) |
| [Remasking discrete diffusion models with inference-time scaling](https://arxiv.org/abs/2503.00307)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/kuleshov-group/remdm?style=flat)](https://github.com/kuleshov-group/remdm) |
| [Path planning for masked diffusion model sampling](https://arxiv.org/abs/2502.03540)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/pengzhangzhi/Path-Planning?style=flat)](https://github.com/pengzhangzhi/Path-Planning) |
| [Think while you generate: Discrete diffusion with planned denoising](https://arxiv.org/abs/2410.06264)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/liusulin/DDPD?style=flat)](https://github.com/liusulin/DDPD) |
| [Accelerating Diffusion LLMs via Adaptive Parallel Decoding](https://arxiv.org/abs/2506.00413)  |  arxiv 2025  | - |
| [Reviving any-subset autoregressive models with principled parallel sampling and speculative decoding](https://arxiv.org/abs/2504.20456)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/gabeguo/any-order-speculative-decoding?style=flat)](https://github.com/gabeguo/any-order-speculative-decoding) |
| [dkv-cache: The cache for diffusion language models](https://arxiv.org/abs/2505.15781)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/horseee/dKV-Cache?style=flat)](https://github.com/horseee/dKV-Cache) |
| [Accelerating diffusion language model inference via efficient kv caching and guided diffusion](https://arxiv.org/abs/2505.21467)  |  arxiv 2025  | - |
| [Esoteric Language Models](https://arxiv.org/abs/2506.01928)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/s-sahoo/Eso-LMs?style=flat)](https://github.com/s-sahoo/Eso-LMs) |
| [Beyond Autoregression: Fast LLMs via Self-Distillation Through Time](https://arxiv.org/abs/2410.21035)  |  ICLR 2025  | - |
| [Cllms: Consistency large language models](https://openreview.net/forum?id=8uzBOVmh8H)  |  ICML 2024  | [![Github](https://img.shields.io/github/stars/hao-ai-lab/Consistency_LLM?style=flat)](https://github.com/hao-ai-lab/Consistency_LLM) |
| [The diffusion duality](https://arxiv.org/abs/2506.10892)  |  ICML 2025  | [![Github](https://img.shields.io/github/stars/s-sahoo/duo?style=flat)](https://github.com/s-sahoo/duo) |
| [d1: Scaling reasoning in diffusion large language models via reinforcement learning](https://arxiv.org/abs/2504.12216)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/dllm-reasoning/d1?style=flat)](https://github.com/dllm-reasoning/d1) |
| [LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models](https://arxiv.org/abs/2505.19223)  |  AAAI 2025  | [![Github](https://img.shields.io/github/stars/ML-GSAI/LLaDA-1.5?style=flat)](https://github.com/ML-GSAI/LLaDA-1.5) |
| [DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/apple/ml-diffucoder?style=flat)](https://github.com/apple/ml-diffucoder) |
| [Scaling diffusion language models via adaptation from autoregressive models](https://arxiv.org/abs/2410.17891)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/HKUNLP/DiffuLLaMA?style=flat)](https://github.com/HKUNLP/DiffuLLaMA) |
| [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/DreamLM/Dream?style=flat)](https://github.com/DreamLM/Dream) |
| [DiffPO: Diffusion-styled Preference Optimization for Inference Time Alignment of Large Language Models](https://arxiv.org/abs/2503.04240)  |  ACL 2025  | [![Github](https://img.shields.io/github/stars/zjuruizhechen/DiffPO?style=flat)](https://github.com/zjuruizhechen/DiffPO) |
| [Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs](https://arxiv.org/abs/2507.18578)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/Feng-Hong/WINO-DLLM?style=flat)](https://github.com/Feng-Hong/WINO-DLLM) |
| [Dream-Coder 7B: An Open Diffusion Language Model for Code](https://arxiv.org/abs/2509.01142)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/DreamLM/Dream-Coder?style=flat)](https://github.com/DreamLM/Dream-Coder) |
| [Spg: Sandwiched policy gradient for masked diffusion language models](https://arxiv.org/abs/2510.09541)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/facebookresearch/SPG?style=flat)](https://github.com/facebookresearch/SPG) |
| [Revolutionizing reinforcement learning framework for diffusion large language models](https://arxiv.org/abs/2509.06949)  |  ICLR 2026  | [![Github](https://img.shields.io/github/stars/Gen-Verse/dLLM-RL?style=flat)](https://github.com/Gen-Verse/dLLM-RL) |
| [Diffusion llms can do faster-than-ar inference via discrete diffusion forcing](https://arxiv.org/abs/2508.09192)  |  ICLR 2026  | [![Github](https://img.shields.io/github/stars/SJTU-DENG-Lab/Discrete-Diffusion-Forcing?style=flat)](https://github.com/SJTU-DENG-Lab/Discrete-Diffusion-Forcing) |
| [WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference](https://arxiv.org/abs/2512.22737)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/Tencent/WeDLM?style=flat)](https://github.com/Tencent/WeDLM) |
| [d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](https://arxiv.org/abs/2512.09675)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/THU-BPM/d-TreeRPO?style=flat)](https://github.com/THU-BPM/d-TreeRPO) |
| [d2: Improved Techniques for Training Reasoning Diffusion Language Models](https://arxiv.org/abs/2509.21474)  |  arxiv 2025  | - |
| [wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](https://arxiv.org/pdf/2507.08838)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/xiaohangt/wd1?style=flat)](https://github.com/xiaohangt/wd1) |
| [Step-Aware Policy Optimization for Reasoning in Diffusion Large Language Models](https://arxiv.org/abs/2510.01544)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/Mid-Push/SAPO-LLaDA?style=flat)](https://github.com/Mid-Push/SAPO-LLaDA) |
| [Improving reasoning for diffusion language models via group diffusion policy optimization](https://arxiv.org/abs/2510.08554)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/kevinrojas1499/GDPO?style=flat)](https://github.com/kevinrojas1499/GDPO) |
| [The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models](https://arxiv.org/abs/2601.15165)  |  arxiv 2026  | [![Github](https://img.shields.io/github/stars/LeapLabTHU/JustGRPO?style=flat)](https://github.com/LeapLabTHU/JustGRPO) |
| [Principled rl for diffusion llms emerges from a sequence-level perspective](https://arxiv.org/abs/2512.03759)  |  ICLR 2026  | [![Github](https://img.shields.io/github/stars/ML-GSAI/ESPO?style=flat)](https://github.com/ML-GSAI/ESPO) |
| [DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](https://arxiv.org/abs/2602.01326)  |  arxiv 2026  | [![Github](https://img.shields.io/github/stars/DreamLM/DreamOn?style=flat)](https://github.com/DreamLM/DreamOn) |
| [SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation](https://arxiv.org/abs/2510.06303)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/JetAstra/SDAR?style=flat)](https://github.com/JetAstra/SDAR) |
| [Lopa: Scaling dllm inference via lookahead parallel decoding](https://arxiv.org/abs/2512.16229)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/SJTU-DENG-Lab/LoPA?style=flat)](https://github.com/SJTU-DENG-Lab/LoPA) |
| [FAST-dLLM V2: Efficient Block-Diffusion LLM](https://arxiv.org/pdf/2509.26328)  |  ICLR 2026  | [![Github](https://img.shields.io/github/stars/NVlabs/Fast-dLLM?style=flat)](https://github.com/NVlabs/Fast-dLLM) |
| [d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation](https://arxiv.org/abs/2601.07568)  |  arxiv 2026  | [![Github](https://img.shields.io/github/stars/hao-ai-lab/d3LLM?style=flat)](https://github.com/hao-ai-lab/d3LLM) |
| [dParallel: Learnable Parallel Decoding for dLLMs](https://arxiv.org/abs/2509.26488)  |  ICLR 2026  | [![Github](https://img.shields.io/github/stars/czg1225/dParallel?style=flat)](https://github.com/czg1225/dParallel) |
| [Diffusion language models know the answer before decoding](https://arxiv.org/abs/2508.19982)  |  ICLR 2026  | [![Github](https://img.shields.io/github/stars/pixeli99/Prophet?style=flat)](https://github.com/pixeli99/Prophet) |
| [Creditdecoding: Accelerating parallel decoding in diffusion large language models with trace credits](https://arxiv.org/abs/2510.06133)  |  arxiv 2025  | - |
| [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/kuleshov-group/bd3lms?style=flat)](https://github.com/kuleshov-group/bd3lms) |
| [Set Block Decoding is a Language Model Inference Accelerator](https://arxiv.org/abs/2509.04185)  |  arxiv 2025  | - |
| [LLaDA-MoE: A Sparse MoE Diffusion Language Model](https://arxiv.org/abs/2509.24389)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/ML-GSAI/LLaDA?style=flat)](https://github.com/ML-GSAI/LLaDA) |
| [dInfer: An Efficient Inference Framework for Diffusion Language Models](https://arxiv.org/abs/2510.08666)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/inclusionAI/dInfer?style=flat)](https://github.com/inclusionAI/dInfer) |
| [ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs](https://arxiv.org/abs/2510.04767)  |  ICLR 2026  | [![Github](https://img.shields.io/github/stars/furiosa-ai/ParallelBench?style=flat)](https://github.com/furiosa-ai/ParallelBench) |
| [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://aclanthology.org/D19-1633/) | EMNLP-IJCNLP 2019 | [![Github](https://img.shields.io/github/stars/facebookresearch/Mask-Predict?style=flat)](https://github.com/facebookresearch/Mask-Predict) |
| [Likelihood-Based Diffusion Language Models](https://arxiv.org/abs/2305.18619) | NeurIPS 2023 | [![Github](https://img.shields.io/github/stars/igul222/plaid?style=flat)](https://github.com/igul222/plaid) |
| [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217) | NeurIPS 2022 | [![Github](https://img.shields.io/github/stars/XiangLi1999/Diffusion-LM?style=flat)](https://github.com/XiangLi1999/Diffusion-LM) |
| [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/) | Technical release 2025 | — |
| [Mercury: Ultra-Fast Language Models Based on Diffusion](https://arxiv.org/abs/2506.17298) | arXiv 2025 | — |
| [LLaDA2.0: Scaling Up Diffusion Language Models to 100B](https://arxiv.org/abs/2512.15745) | arXiv 2025 | [![Github](https://img.shields.io/github/stars/inclusionAI/LLaDA2.X?style=flat)](https://github.com/inclusionAI/LLaDA2.X) |
| [LLaDA2.1: Speeding Up Text Diffusion via Token Editing](https://arxiv.org/abs/2602.08676) | arXiv 2026 | [![Github](https://img.shields.io/github/stars/inclusionAI/LLaDA2.X?style=flat)](https://github.com/inclusionAI/LLaDA2.X) |
| [DiffusionGemma Model Card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card) | Model card 2026 | — |
| [Diffusion Beats Autoregressive in Data-Constrained Settings](https://arxiv.org/abs/2507.15857) | arXiv 2025 | — |
| [Fast-decoding diffusion language models via progress-aware confidence schedules](https://aclanthology.org/2026.findings-acl.1782/) | Findings ACL 2026 | — |
| [From Bits to Rounds: Parallel Decoding with Exploration for Diffusion Language Models](https://arxiv.org/abs/2511.21103) | arXiv 2025 | — |
| [Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding](https://arxiv.org/abs/2509.25188) | arXiv 2025 | — |
| [Introspective Diffusion Language Models](https://arxiv.org/abs/2604.11035) | arXiv 2026 | — |
| [ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](https://arxiv.org/abs/2512.13586) | ICLR 2026 | [![Github](https://img.shields.io/github/stars/ML-GSAI/ReFusion?style=flat)](https://github.com/ML-GSAI/ReFusion) |
| [LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning](https://arxiv.org/abs/2505.16933) | arXiv 2025 | — |
| [Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](https://arxiv.org/abs/2505.16990) | arXiv 2025 | [![Github](https://img.shields.io/github/stars/yu-rp/Dimple?style=flat)](https://github.com/yu-rp/Dimple) |
| [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219) | arXiv 2023 | — |
| [Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data](https://arxiv.org/abs/2406.03736) | ICLR 2025 | [![Github](https://img.shields.io/github/stars/ML-GSAI/RADD?style=flat)](https://github.com/ML-GSAI/RADD) |
| [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) | NeurIPS 2024 | [![Github](https://img.shields.io/github/stars/kuleshov-group/mdlm?style=flat)](https://github.com/kuleshov-group/mdlm) |
| [Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514) | ICLR 2025 | [![Github](https://img.shields.io/github/stars/ML-GSAI/SMDM?style=flat)](https://github.com/ML-GSAI/SMDM) |
| [Diffusion Language Models are Super Data Learners](https://arxiv.org/abs/2511.03276) | arXiv 2025 | — |
| [Locally Confident, Globally Stuck: The Quality-Exploration Dilemma in Diffusion Language Models](https://arxiv.org/abs/2604.00375) | arXiv 2026 | — |
| [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) | NeurIPS 2021 | — |
| [Accelerating Transformer Inference for Translation via Parallel Decoding](https://arxiv.org/abs/2305.10427) | arXiv 2023 | — |
| [CtrlDiff: Boosting Large Diffusion Language Models with Dynamic Block Prediction and Controllable Generation](https://arxiv.org/abs/2505.14455) | arXiv 2025 | — |
| [dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](https://arxiv.org/abs/2506.06295) | arXiv 2025 | [![Github](https://img.shields.io/github/stars/maomaocun/dLLM-cache?style=flat)](https://github.com/maomaocun/dLLM-cache) |
| [Attention Is All You Need for KV Cache in Diffusion LLMs](https://arxiv.org/abs/2510.14973) | ICLR 2026 | [![Github](https://img.shields.io/github/stars/VILA-Lab/Elastic-Cache?style=flat)](https://github.com/VILA-Lab/Elastic-Cache) |
| [Generation Order and Parallel Decoding in Masked Diffusion Models: An Information-Theoretic Perspective](https://arxiv.org/abs/2602.00286) | arXiv 2026 | — |
| [Theoretical Benefit and Limitation of Diffusion Language Model](https://arxiv.org/abs/2502.09622) | arXiv 2025 | — |
| [Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models](https://arxiv.org/abs/2602.19619) | arXiv 2026 | — |
| [Parallelism and Generation Order in Masked Diffusion Language Models: Limits Today, Potential Tomorrow](https://arxiv.org/abs/2601.15593) | arXiv 2026 | — |

##### Related Foundations

The following works underpin the formulations, decoding strategies, or training objectives discussed in this section and the appendices. Image-generation methods and general training objectives are background sources, not additional evidence of parallel text-generation speedups.

| Paper | Venue | Role |
| --- | :---: | --- |
| [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200) | CVPR 2022 | Masked image-generation precursor |
| [Improved Masked Image Generation with Token-Critic](https://arxiv.org/abs/2209.04439) | ECCV 2022 | Token-selection precursor |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | ICML 2023 | Step-reduction foundation |
| [Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189) | arXiv 2023 | Step-reduction foundation |
| [Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models](https://arxiv.org/abs/2410.11081) | arXiv 2024 | Step-reduction foundation |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | NeurIPS 2020 | Diffusion formulation |
| [Deep unsupervised learning using nonequilibrium thermodynamics](https://proceedings.mlr.press/v37/sohl-dickstein15.html) | ICML 2015 | Diffusion formulation |
| [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | ICLR 2021 | Continuous-time formulation |
| [Approximate accelerated stochastic simulation of chemically reacting systems](https://doi.org/10.1063/1.1378322) | J. Chem. Phys. 2001 | Tau-leaping foundation |
| [Concrete Score Matching: Generalized Score Matching for Discrete Data](https://arxiv.org/abs/2211.00802) | NeurIPS 2022 | Discrete score matching |
| [A deep and tractable density estimator](https://proceedings.mlr.press/v32/uria14.html) | ICML 2014 | Order-agnostic modeling |
| [Training and Inference on Any-Order Autoregressive Models the Right Way](https://arxiv.org/abs/2205.13554) | NeurIPS 2022 | Any-order modeling |
| [Autoregressive Diffusion Models](https://arxiv.org/abs/2110.02037) | arXiv 2021 | Any-order / diffusion connection |
| [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) | NeurIPS 2019 | Permutation language modeling |
| [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | NeurIPS 2023 | Preference-optimization foundation |
| [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) | arXiv 2024 | GRPO training foundation |

#### Edit-Based Refinement

Following the survey, this section covers discrete editing, continuous optimization, hybrid refinement, and masked-edit refinement, together with supporting training methods. Hybrid self-correction may use AR generators internally; iterative refinement alone does not establish a parallel speedup.

| Paper | Venue | Code |
| --- | :---: | :---: |
| [Insertion transformer: Flexible sequence generation via insertion operations](https://proceedings.mlr.press/v97/stern19a.html)  |  ICML 2019  | - |
| [Levenshtein transformer](https://proceedings.neurips.cc/paper/2019/hash/675f9820626f5bc0afb47b57890b466e-Abstract.html)  |  NeurIPS 2019  | [![Github](https://img.shields.io/github/stars/pytorch/fairseq?style=flat)](https://github.com/pytorch/fairseq) |
| [EDITOR: An edit-based transformer with repositioning for neural machine translation with soft lexical constraints](https://direct.mit.edu/tacl/article-abstract/doi/10.1162/tacl_a_00368/98622)  |  TACL 2021  | [![Github](https://img.shields.io/github/stars/Izecson/fairseq-editor?style=flat)](https://github.com/Izecson/fairseq-editor) |
| [FELIX: Flexible Text Editing Through Tagging and Insertion](https://arxiv.org/abs/2003.10687)  |  EMNLP 2020  | - |
| [Levenshtein OCR](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_19)  |  ECCV 2022  | [![Github](https://img.shields.io/github/stars/AlibabaResearch/AdvancedLiterateMachinery?style=flat)](https://github.com/AlibabaResearch/AdvancedLiterateMachinery) |
| [FastCorrect: Fast Error Correction with Edit Alignment for Automatic Speech Recognition](https://proceedings.neurips.cc/paper_files/paper/2021/hash/b597460c506e8e35fb0cc1c1905dd3bc-Abstract.html)  |  NeurIPS 2021  | [![Github](https://img.shields.io/github/stars/microsoft/NeuralSpeech?style=flat)](https://github.com/microsoft/NeuralSpeech) |
| [Non-autoregressive Text Editing with Copy-aware Latent Alignments](https://arxiv.org/abs/2310.07821)  |  EMNLP 2023  | [![Github](https://img.shields.io/github/stars/yzhangcs/ctc-copy?style=flat)](https://github.com/yzhangcs/ctc-copy) |
| [Reinforcement Learning for Edit-Based Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/2405.01280)  |  NAACL-SRW 2024  | - |
| [Summarizing Like Human: Edit-Based Text Summarization with Keywords](https://link.springer.com/chapter/10.1007/978-3-031-72350-6_23)  |  ICANN 2024  | - |
| [Deterministic non-autoregressive neural sequence modeling by iterative refinement](https://arxiv.org/abs/1802.06901)  |  EMNLP 2018  | [![Github](https://img.shields.io/github/stars/nyu-dl/dl4mt-nonauto?style=flat)](https://github.com/nyu-dl/dl4mt-nonauto) |
| [Flowseq: Non-autoregressive conditional sequence generation with generative flow](https://arxiv.org/abs/1909.02480)  |  EMNLP 2019  | [![Github](https://img.shields.io/github/stars/XuezheMax/flowseq?style=flat)](https://github.com/XuezheMax/flowseq) |
| [Latent-variable non-autoregressive neural machine translation with deterministic inference using a delta posterior](https://aaai.org/ojs/index.php/AAAI/article/view/6413)  |  AAAI 2020  | [![Github](https://img.shields.io/github/stars/zomux/lanmt?style=flat)](https://github.com/zomux/lanmt) |
| [Iterative Refinement in the Continuous Space for Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/2009.07177)  |  EMNLP 2020  | [![Github](https://img.shields.io/github/stars/zomux/lanmt-ebm?style=flat)](https://github.com/zomux/lanmt-ebm) |
| [Non-autoregressive machine translation with auxiliary regularization](https://ojs.aaai.org/index.php/AAAI/article/view/4476)  |  AAAI 2019  | - |
| [Imitation learning for non-autoregressive neural machine translation](https://arxiv.org/abs/1906.02041)  |  ACL 2019  | - |
| [An imitation learning curriculum for text editing with non-autoregressive models](https://arxiv.org/abs/2203.09486)  |  ACL 2022  | [![Github](https://img.shields.io/github/stars/sweta20/EditingCL?style=flat)](https://github.com/sweta20/EditingCL) |
| [Fast structured decoding for sequence models](https://proceedings.neurips.cc/paper/2019/hash/74563ba21a90da13dacf2a73e3ddefa7-Abstract.html)  |  NeurIPS 2019  | [![Github](https://img.shields.io/github/stars/Edward-Sun/structured-nart?style=flat)](https://github.com/Edward-Sun/structured-nart) |
| [An EM approach to non-autoregressive conditional sequence generation](http://proceedings.mlr.press/v119/sun20c.html)  |  ICML 2020  | - |
| [Imputer: Sequence modelling via imputation and dynamic programming](http://proceedings.mlr.press/v119/chan20b.html)  |  ICML 2020  | [![Github](https://img.shields.io/github/stars/rosinality/imputer-pytorch?style=flat)](https://github.com/rosinality/imputer-pytorch) |
| [Align-Refine: Non-Autoregressive Speech Recognition via Iterative Realignment](https://arxiv.org/abs/2010.14233)  |  NAACL 2021  | [![Github](https://img.shields.io/github/stars/amazon-research/align-refine?style=flat)](https://github.com/amazon-research/align-refine) |
| [Learning to rewrite for non-autoregressive neural machine translation](https://aclanthology.org/2021.emnlp-main.265)  |  EMNLP 2021  | [![Github](https://img.shields.io/github/stars/xwgeng/RewriteNAT?style=flat)](https://github.com/xwgeng/RewriteNAT) |
| [RenewNAT: renewing potential translation for non-autoregressive transformer](https://arxiv.org/abs/2303.07665)  |  AAAI 2023  | - |
| [Learning to recover from multi-modality errors for non-autoregressive neural machine translation](https://arxiv.org/abs/2006.05165)  |  ACL 2020  | [![Github](https://img.shields.io/github/stars/ranqiu92/RecoverSAT?style=flat)](https://github.com/ranqiu92/RecoverSAT) |
| [Hybrid-regressive neural machine translation](https://arxiv.org/abs/2210.10416)  |  ICLR 2023  | - |
| [Iterative Translation Refinement with Large Language Models](https://arxiv.org/abs/2306.03856)  |  EAMT 2024  | - |
| [IterGen: Iterative Semantic-aware Structured LLM Generation with Backtracking](https://arxiv.org/abs/2410.07295)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/structuredllm/itergen?style=flat)](https://github.com/structuredllm/itergen) |
| [Rejuvenating low-frequency words: Making the most of parallel data in non-autoregressive translation](https://arxiv.org/abs/2106.00903)  |  ACL 2021  | [![Github](https://img.shields.io/github/stars/alphadl/RLFW-NAT?style=flat)](https://github.com/alphadl/RLFW-NAT) |
| [Understanding and Improving Lexical Choice in Non-Autoregressive Translation](https://arxiv.org/abs/2012.14583)  |  ICLR 2021  | [![Github](https://img.shields.io/github/stars/alphadl/LCNAT?style=flat)](https://github.com/alphadl/LCNAT) |
| [SlotRefine: A fast non-autoregressive model for joint intent detection and slot filling](https://arxiv.org/abs/2010.02693)  |  EMNLP 2020  | [![Github](https://img.shields.io/github/stars/moore3930/SlotRefine?style=flat)](https://github.com/moore3930/SlotRefine) |
| [Non-autoregressive dialog state tracking](https://arxiv.org/abs/2002.08024)  |  ICLR 2020  | [![Github](https://img.shields.io/github/stars/henryhungle/NADST?style=flat)](https://github.com/henryhungle/NADST) |
| [Unleashing the True Potential of LLMs: A Feedback-Triggered Self-Correction with Long-Term Multipath Decoding](https://arxiv.org/abs/2509.07676)  |  arxiv 2025  | - |
| [ProRefine: Inference-Time Prompt Refinement with Textual Feedback](https://arxiv.org/abs/2506.05305)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/deepakpandita57/ProRefine_public?style=flat)](https://github.com/deepakpandita57/ProRefine_public) |
| [A Probabilistic Inference Scaling Theory for LLM Self-Correction](https://arxiv.org/abs/2508.16456)  |  EMNLP 2025  | [![Github](https://img.shields.io/github/stars/probabilistic-inference-scaling/probabilistic-inference-scaling?style=flat)](https://github.com/probabilistic-inference-scaling/probabilistic-inference-scaling) |
| [Self-Refine: Iterative Refinement with Self-Feedback](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)  |  NeurIPS 2023  | [![Github](https://img.shields.io/github/stars/madaan/self-refine?style=flat)](https://github.com/madaan/self-refine) |
| [Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model in Code Generation](https://aclanthology.org/2026.acl-long.165/) | ACL 2026 | — |
| [Edit-Based Refinement for Parallel Masked Diffusion Language Models](https://arxiv.org/abs/2605.09603) | arXiv 2026 | — |
| [An Empirical Study of Iterative Refinements for Non-autoregressive Translation](https://aclanthology.org/2025.acl-long.1443/) | ACL 2025 | — |
| [Towards More Efficient Insertion Transformer with Fractional Positional Encoding](https://aclanthology.org/2023.eacl-main.115/) | EACL 2023 | — |

## Task-Level Parallelism

Task-level methods expose independent content units, subplans, or reasoning branches. Evaluation should distinguish end-to-end latency, aggregate computation, concurrent resources, and synchronization/merging overhead.

### Content Decomposition

#### Query Decomposition

Decomposed Prompting and Least-to-Most Prompting are decomposition precursors; dependent subtasks remain sequential unless their dependencies permit parallel execution.

| Paper | Venue | Code |
| --- | :---: | :---: |
| [PARALLELPROMPT: Extracting Parallelism from Large Language Model Queries](https://arxiv.org/abs/2506.18728)  |  arxiv 2025  | - |
| [Decomposed Prompting: A Modular Approach for Solving Complex Tasks](https://arxiv.org/abs/2210.02406) | ICLR 2023 | [![Github](https://img.shields.io/github/stars/allenai/DecomP?style=flat)](https://github.com/allenai/DecomP) |
| [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625) | ICLR 2023 | — |

#### Structure Planning

SoT expands skeleton points concurrently. WritingPath, Plan-and-Write, and outline-conditioned generation are planning precursors, not evidence of parallel execution.

| Paper | Venue | Code |
| --- | :---: | :---: |
| [Navigating the Path of Writing: Outline-guided Text Generation with Large Language Models](https://arxiv.org/abs/2404.13919)  |  NAACL 2025  | - |
| [Skeleton-of-thought: Prompting llms for efficient parallel generation](https://arxiv.org/abs/2307.15337)  |  ICLR 2024  | [![Github](https://img.shields.io/github/stars/imagination-research/sot?style=flat)](https://github.com/imagination-research/sot) |
| [Plan-And-Write: Towards Better Automatic Storytelling](https://arxiv.org/abs/1811.05701) | AAAI 2019 | — |
| [Advancing Precise Outline-Conditioned Text Generation with Task Duality and Explicit Outline Control](https://aclanthology.org/2024.eacl-long.145/) | EACL 2024 | — |

### Process Orchestration

#### Plan Execution

| Paper | Venue | Code |
| --- | :---: | :---: |
| [SPRINT: Enabling Interleaved Planning and Parallelized Execution in Reasoning Models](https://arxiv.org/abs/2506.05745)  |  arxiv 2025  | - |
| [Decompose, Plan in Parallel, and Merge: A Novel Paradigm for Large Language Models based Planning with Multiple Constraints](https://arxiv.org/abs/2506.02683) | arXiv 2025 | — |
| [Optimizing Sequential Multi-Step Tasks with Parallel LLM Agents](https://arxiv.org/abs/2507.08944) | arXiv 2025 | — |
| [AgentSPEX: An Agent SPecification and EXecution Language](https://arxiv.org/abs/2604.13346) | arXiv 2026 | — |

#### Branch Reasoning

| Paper | Venue | Code |
| --- | :---: | :---: |
| [Learning to Keep a Promise: Scaling Language Model Decoding Parallelism with Learned Asynchronous Decoding (PASTA)](https://proceedings.mlr.press/v267/jin25a.html) | ICML 2025 | — |
| [APAR: LLMs Can Do Auto-Parallel Auto-Regressive Decoding](https://arxiv.org/abs/2401.06761) | arXiv 2024 | — |
| [Learning Adaptive Parallel Reasoning with Language Models](https://arxiv.org/abs/2504.15466) | arXiv 2025 | — |
| [Accelerate Parallelizable Reasoning via Parallel Decoding within One Sequence](https://aclanthology.org/2025.emnlp-main.457/) | EMNLP 2025 | — |
| [Training Large Language Models To Reason In Parallel With Global Forking Tokens](https://arxiv.org/abs/2510.05132) | ICLR 2026 | — |
| [Towards Direct Latent-Space Synthesis for Parallel Branches in LLM-Agent Workflows](https://arxiv.org/abs/2606.14672) | arXiv 2026 | — |

## Complementary Acceleration

These techniques are discussed in the manuscript's composability appendix, not treated as additional parallel-generation paradigms. AWQ and GPTQ concern model compression, FlashAttention-2 concerns attention kernels, and LazyMAR is a visual-generation caching example rather than a text-decoding result.

| Paper | Venue | Code |
| --- | :---: | :---: |
| [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) | MLSys 2024 | [![Github](https://img.shields.io/github/stars/mit-han-lab/llm-awq?style=flat)](https://github.com/mit-han-lab/llm-awq) |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) | ICLR 2023 | [![Github](https://img.shields.io/github/stars/IST-DASLab/gptq?style=flat)](https://github.com/IST-DASLab/gptq) |
| [LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](https://arxiv.org/abs/2503.12450) | ICCV 2025 | [![Github](https://img.shields.io/github/stars/feihongyan1/LazyMAR?style=flat)](https://github.com/feihongyan1/LazyMAR) |
| [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) | ICLR 2024 | [![Github](https://img.shields.io/github/stars/Dao-AILab/flash-attention?style=flat)](https://github.com/Dao-AILab/flash-attention) |

## Related Surveys

The entries and main scopes below match Table 1 of the revised manuscript. Scopes describe each review's emphasis and may overlap. Publication years refer to the cited versions.

| Reference | Year | Survey | Main scope |
| --- | :---: | --- | --- |
| Xia et al. | 2024 | [Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding](https://aclanthology.org/2024.findings-acl.456/) | Speculative decoding: drafter selection and verification |
| Zhang et al. | 2024 | [Beyond the Speculative Game: A Survey of Speculative Execution in Large Language Models](https://arxiv.org/abs/2404.14897) | Speculative execution in LLM inference |
| Ryu and Kim | 2024 | [Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding](https://arxiv.org/abs/2411.13157) | Speculative decoding: draft- and model-centric methods |
| Hu et al. | 2025 | [Mitigating Sequential Dependencies: A Survey of Algorithms and Systems for Generation-Refinement Frameworks in Autoregressive Models](https://aclanthology.org/2025.findings-emnlp.716/) | Generation–refinement frameworks: algorithms and systems |
| Khoshnoodi et al. | 2024 | [A Comprehensive Survey of Accelerated Generation Techniques in Large Language Models](https://arxiv.org/abs/2405.13019) | Accelerated generation: speculation, early exiting, and NAR methods |
| Wan et al. | 2024 | [Efficient Large Language Models: A Survey](https://openreview.net/forum?id=bsCCJHbO8A) | LLM efficiency: model-, data-, and framework-centric methods |
| Li et al. | 2023 | [Diffusion Models for Non-autoregressive Text Generation: A Survey](https://www.ijcai.org/proceedings/2023/750) | Diffusion models for non-autoregressive text generation |
| Yi et al. | 2024 | [Diffusion Models in Text Generation: A Survey](https://peerj.com/articles/cs-1905/) | Diffusion-based text generation across task settings |
| Yu et al. | 2025 | [Discrete Diffusion in Large Language and Multimodal Models: A Survey](https://arxiv.org/abs/2506.13759) | Discrete diffusion language and multimodal models |
| Li et al. | 2025 | [A Survey on Diffusion Language Models](https://arxiv.org/abs/2508.10875) | Diffusion language models: training, inference, and applications |
