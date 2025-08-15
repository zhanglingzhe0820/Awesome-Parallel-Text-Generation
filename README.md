# Awesome-Parallel-Text-Generation

## Our Survey
[A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](https://arxiv.org/abs/2508.08712)

The first comprehensive survey for Parallel Text Generation Methods. [[PDF](https://arxiv.org/pdf/2508.08712)]

## Methodology

### AR-Based

#### Draft-and-Verify

| Paper      | Venue       | Code   |  
|-----------|:----------------:|:--------------:| 
| [Adaptive Draft-Verification for Efficient Large Language Model Decoding](https://ojs.aaai.org/index.php/AAAI/article/view/34647)  |  AAAI 2025  | [![Github](https://img.shields.io/github/stars/liuxukun2000/Adaptix?style=flat)](https://github.com/liuxukun2000/Adaptix) | 
| [Speculative Decoding with Big Little Decoder](https://arxiv.org/abs/2302.07863)  |  NeurIPS 2023  | [![Github](https://img.shields.io/github/stars/kssteven418/BigLittleDecoder?style=flat)](https://github.com/kssteven418/BigLittleDecoder) | 
| [Block Verification Accelerates Speculative Decoding](https://arxiv.org/abs/2403.10444)  |  ICLR 2025  | - | 
| [Cascade speculative drafting for even faster llm inference](https://arxiv.org/abs/2312.11462)  |  NeurIPS 2023  | [![Github](https://img.shields.io/github/stars/lfsszd/CS-Drafting?style=flat)](https://github.com/lfsszd/CS-Drafting) | 
| [Dynamic Depth Decoding: Faster Speculative Decoding for LLMs](https://arxiv.org/abs/2409.00142)  |  arxiv 2024  | - |
| [Distillspec: Improving speculative decoding via knowledge distillation](https://arxiv.org/abs/2310.08461)  |  ICLR 2024  | - |
| [Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168)  |  ACL 2024  | [![Github](https://img.shields.io/github/stars/dilab-zju/self-speculative-decoding?style=flat)](https://github.com/dilab-zju/self-speculative-decoding) |
| [Dynamic-Width Speculative Beam Decoding for Efficient LLM Inference](https://arxiv.org/abs/2409.16560)  |  AAAI 2025  | [![Github](https://img.shields.io/github/stars/ZongyueQin/DSBD?style=flat)](https://github.com/ZongyueQin/DSBD) |
| [DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure](https://arxiv.org/abs/2410.11744)  |  WWW 2025  | - |
| [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)  |  ICML 2024  | [![Github](https://img.shields.io/github/stars/SafeAILab/EAGLE?style=flat)](https://github.com/SafeAILab/EAGLE) |
| [Eagle-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)  |  EMNLP 2024  | [![Github](https://img.shields.io/github/stars/SafeAILab/EAGLE?style=flat)](https://github.com/SafeAILab/EAGLE) |
| [Speculative Decoding via Early-Exiting for Faster LLM Inference with Thompson Sampling Control Mechanism](https://arxiv.org/abs/2406.03853)  |  ACL 2024  | - |
| [Falcon: Faster and Parallel Inference of Large Language Models through Enhanced Semi-Autoregressive Drafting and Custom-designed Decoding Tree](https://ojs.aaai.org/index.php/AAAI/article/view/34566)  |  AAAI 2025  | [![Github](https://img.shields.io/github/stars/Bestpay-inc/Falcon?style=flat)](https://github.com/Bestpay-inc/Falcon) |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)  |  ICML 2023  | [![Github](https://img.shields.io/github/stars/romsto/Speculative-Decoding?style=flat)](https://github.com/romsto/Speculative-Decoding) |
| [Graph-Structured Speculative Decoding](https://arxiv.org/abs/2407.16207)  |  ACL 2024  | [![Github](https://img.shields.io/github/stars/gzhch/gsd?style=flat)](https://github.com/gzhch/gsd) |
| [Learning Harmonized Representations for Speculative Sampling](https://arxiv.org/abs/2408.15766)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/HArmonizedSS/HASS?style=flat)](https://github.com/HArmonizedSS/HASS) |
| [Hydra: Sequentially-dependent draft heads for medusa decoding](https://arxiv.org/abs/2302.07863)  |  COLM 2024  | [![Github](https://img.shields.io/github/stars/zankner/Hydra?style=flat)](https://github.com/zankner/Hydra) |
| [Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment](https://arxiv.org/abs/2501.19309)  |  ICLR 2025  | - |
| [Kangaroo: Lossless self-speculative decoding via double early exiting](https://openreview.net/forum?id=lT3oc04mDp)  |  NeurIPS 2024  | [![Github](https://img.shields.io/github/stars/Equationliu/Kangaroo?style=flat)](https://github.com/Equationliu/Kangaroo) |
| [Layer-skip: Enabling early-exit inference and self-speculative decoding](https://arxiv.org/abs/2404.16710)  |  ACL 2024  | [![Github](https://img.shields.io/github/stars/facebookresearch/LayerSkip?style=flat)](https://github.com/facebookresearch/LayerSkip) |
| [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)  |  PMLR 2024  | [![Github](https://img.shields.io/github/stars/FasterDecoding/Medusa?style=flat)](https://github.com/FasterDecoding/Medusa) |
| [Mixture of Attentions for Speculative Decoding](https://arxiv.org/abs/2410.03804)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/huawei-noah/HEBO?style=flat)](https://github.com/huawei-noah/HEBO) |
| [Optimized multi-token joint decoding with auxiliary model for llm inference](https://arxiv.org/abs/2407.09722)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/ZongyueQin/MTAD?style=flat)](https://github.com/ZongyueQin/MTAD) |
| [A Drop-in Solution for On-the-fly Adaptation of Speculative Decoding in Large Language Models](https://aclanthology.org/2025.acl-long.482)  |  ACL 2025  | - |
| [OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure](https://arxiv.org/abs/2406.17276)  |  TACL 2025  | [![Github](https://img.shields.io/github/stars/Jikai0Wang/OPT-Tree?style=flat)](https://github.com/Jikai0Wang/OPT-Tree) |
| [Ouroboros: Speculative Decoding with Large Model Enhanced Drafting](https://arxiv.org/abs/2402.13720)  |  EMNLP 2024  | [![Github](https://img.shields.io/github/stars/thunlp/Ouroboros?style=flat)](https://github.com/thunlp/Ouroboros) |
| [Online Speculative Decoding](https://arxiv.org/abs/2310.07177)  |  ICML 2024  | [![Github](https://img.shields.io/github/stars/LiuXiaoxuanPKU/OSD?style=flat)](https://github.com/LiuXiaoxuanPKU/OSD) |
| [Pass: Parallel speculative sampling](https://arxiv.org/abs/2311.13581)  |  NeurIPS-ENLSP 2023  | - |
| [Parallel Speculative Decoding with Adaptive Draft Length](https://arxiv.org/abs/2408.11850)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/smart-lty/ParallelSpeculativeDecoding?style=flat)](https://github.com/smart-lty/ParallelSpeculativeDecoding) |
| [PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation](https://arxiv.org/abs/2407.11798)  |  SC 2024  | [![Github](https://img.shields.io/github/stars/AutonomicPerfectionist/PipeInfer?style=flat)](https://github.com/AutonomicPerfectionist/PipeInfer) |
| [Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding](https://arxiv.org/abs/2307.05908)  |  TMLR 2024  | - |
| [ProPD: Dynamic Token Tree Pruning and Generation for LLM Parallel Decoding](https://arxiv.org/abs/2402.13485)  |  ICCAD 2024  | - |
| [REST: Retrieval-based speculative decoding](https://arxiv.org/abs/2311.08252)  |  NAACL 2024  | [![Github](https://img.shields.io/github/stars/FasterDecoding/REST?style=flat)](https://github.com/FasterDecoding/REST) |
| [Recursive Speculative Decoding: Accelerating LLM Inference via Sampling without Replacement](https://arxiv.org/abs/2402.14160)  |  ICLR-LLMA 2024  | - |
| [Sequoia: Scalable, robust, and hardware-aware speculative decoding](https://arxiv.org/abs/2402.12374)  |  arxiv 2024  | [![Github](https://img.shields.io/github/stars/Infini-AI-Lab/Sequoia?style=flat)](https://github.com/Infini-AI-Lab/Sequoia) |
| [Generation meets verification: Accelerating large language model inference with smart parallel auto-correct decoding](https://arxiv.org/abs/2402.11809)  |  ACL 2024  | [![Github](https://img.shields.io/github/stars/cteant/SPACE?style=flat)](https://github.com/cteant/SPACE) |
| [Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation](https://arxiv.org/abs/2307.02150)  |  EMNLP 2023  | [![Github](https://img.shields.io/github/stars/hemingkx/SpecDec?style=flat)](https://github.com/hemingkx/SpecDec) |
| [Specdec++: Boosting speculative decoding via adaptive candidate lengths](https://arxiv.org/abs/2405.19715)  |  COLM 2025  | [![Github](https://img.shields.io/github/stars/Kaffaljidhmah2/SpecDec_pp?style=flat)](https://github.com/Kaffaljidhmah2/SpecDec_pp) |
| [Specinfer: Accelerating generative large language model serving with tree-based speculative inference and verification](https://arxiv.org/abs/2305.09781)  |  ASPLOS 2024  | [![Github](https://img.shields.io/github/stars/flexflow/FlexFlow?style=flat)](https://github.com/flexflow/FlexFlow) |
| [SpecTr: Fast Speculative Decoding via Optimal Transport](https://arxiv.org/abs/2310.15141)  |  NeurIPS 2023  | - |
| [Speed: speculative pipelined execution for efficient decoding](https://arxiv.org/abs/2310.12072)  |  NeurIPS-ENLSP 2023  | - |
| [Swift: On-the-fly self-speculative decoding for llm inference acceleration](https://arxiv.org/abs/2410.06916)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/hemingkx/SWIFT?style=flat)](https://github.com/hemingkx/SWIFT) |

#### Decomposition-and-Fill

#### Multiple Token Prediction

### Non-AR-Based

#### One-Shot Generation

#### Masked Generation

#### Edit-Based Refinement

More details coming soon...