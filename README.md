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
| [SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning](https://arxiv.org/abs/2504.07891)  |  NeurIPS 2025  | [![Github](https://img.shields.io/github/stars/ruipeterpan/specreason?style=flat)](https://github.com/ruipeterpan/specreason) |
| [Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs](https://arxiv.org/abs/2512.20573)  |  arXiv 2025  | [![Github](https://img.shields.io/github/stars/ruipeterpan/failfast?style=flat)](https://github.com/ruipeterpan/failfast) |


#### Decomposition-and-Fill

| Paper      | Venue       | Code   |  
|-----------|:----------------:|:--------------:| 
| [PARALLELPROMPT: Extracting Parallelism from Large Language Model Queries](https://arxiv.org/abs/2506.18728)  |  arxiv 2025  | - |
| [Falcon: Faster and parallel inference of large language models through enhanced semi-autoregressive drafting and custom-designed decoding tree](https://ojs.aaai.org/index.php/AAAI/article/view/34566)  |  AAAI 2025  | [![Github](https://img.shields.io/github/stars/Bestpay-inc/Falcon?style=flat)](https://github.com/Bestpay-inc/Falcon) | 
| [Navigating the Path of Writing: Outline-guided Text Generation with Large Language Models](https://arxiv.org/abs/2404.13919)  |  NAACL 2025  | - | 
| [Skeleton-of-thought: Prompting llms for efficient parallel generation](https://arxiv.org/abs/2307.15337)  |  ICLR 2024  | [![Github](https://img.shields.io/github/stars/imagination-research/sot?style=flat)](https://github.com/imagination-research/sot) | 
| [SPRINT: Enabling Interleaved Planning and Parallelized Execution in Reasoning Models](https://arxiv.org/abs/2506.05745)  |  arxiv 2025  | - | 

#### Multiple Token Prediction

| Paper      | Venue       | Code   |  
|-----------|:----------------:|:--------------:| 
| [L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](https://arxiv.org/abs/2505.17505)  |  arxiv 2025  | - | 
| [On multi-token prediction for efficient LLM inference](https://arxiv.org/abs/2502.09419)  |  arxiv 2025  | - | 
| [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)  |  ICML 2024  | [![Github](https://img.shields.io/github/stars/FasterDecoding/Medusa?style=flat)](https://github.com/FasterDecoding/Medusa) | 
| [Multi-Token Prediction Needs Registers](https://arxiv.org/abs/2505.10518)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/nasosger/MuToR?style=flat)](https://github.com/nasosger/MuToR) | 
| [Blockwise Parallel Decoding for Deep Autoregressive Models](https://proceedings.neurips.cc/paper/2018/hash/c4127b9194fe8562c64dc0f5bf2c93bc-Abstract.html)  |  NeurIPS 2018  | - | 
| [Pass: Parallel speculative sampling](https://arxiv.org/abs/2311.13581)  |  NeurIPS-ENLSP 2023  | - | 
| [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)  |  ICML 2024  | [![Github](https://img.shields.io/github/stars/SafeAILab/EAGLE?style=flat)](https://github.com/SafeAILab/EAGLE) | 
| [Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential](https://arxiv.org/abs/2507.11851)  |  arxiv 2025  | - | 
| [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063)  |  EMNLP 2020  | [![Github](https://img.shields.io/github/stars/microsoft/ProphetNet?style=flat)](https://github.com/microsoft/ProphetNet) | 
| [Better & faster large language models via multi-token prediction](https://arxiv.org/abs/2404.19737)  |  ICML 2024  | - | 
| [Deepseek-v3 technical report](https://arxiv.org/abs/2412.19437)  |  arxiv 2024  | [![Github](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-V3?style=flat)](https://github.com/deepseek-ai/DeepSeek-V3) | 
| [MiMo: Unlocking the Reasoning Potential of Language Model--From Pretraining to Posttraining](https://arxiv.org/abs/2505.07608)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/xiaomimimo/MiMo?style=flat)](https://github.com/xiaomimimo/MiMo) | 

### Non-AR-Based

#### One-Shot Generation

| Paper      | Venue       | Code   |  
|-----------|:----------------:|:--------------:| 
| [Non-autoregressive neural machine translation](https://arxiv.org/abs/1711.02281)  |  ICLR 2018  | [![Github](https://img.shields.io/github/stars/salesforce/nonauto-nmt?style=flat)](https://github.com/salesforce/nonauto-nmt) | 
| [End-to-end non-autoregressive neural machine translation with connectionist temporal classification](https://arxiv.org/abs/1811.04719)  |  EMNLP 2018  |  | 
| [Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement](https://arxiv.org/abs/1802.06901)  |  EMNLP 2018  | [![Github](https://img.shields.io/github/stars/nyu-dl/dl4mt-nonauto?style=flat)](https://github.com/nyu-dl/dl4mt-nonauto) | 
| [Lava nat: A non-autoregressive translation model with look-around decoding and vocabulary attention](https://arxiv.org/abs/2002.03084)  |  arxiv 2025  | - | 
| [AligNART: Non-autoregressive neural machine translation by jointly learning to estimate alignment and translate](https://arxiv.org/abs/2109.06481)  |  EMNLP 2021  | - | 
| [Guiding non-autoregressive neural machine translation decoding with reordering information](https://ojs.aaai.org/index.php/AAAI/article/view/17618)  |  AAAI 2021  | [![Github](https://img.shields.io/github/stars/ranqiu92/ReorderNAT?style=flat)](https://github.com/ranqiu92/ReorderNAT) | 
| [Non-monotonic latent alignments for ctc-based non-autoregressive machine translation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/35f805e65c77652efa731edc10c8e3a6-Abstract-Conference.html)  |  NeurIPS 2022  | [![Github](https://img.shields.io/github/stars/ictnlp/NMLA-NAT?style=flat)](https://github.com/ictnlp/NMLA-NAT) | 
| [DePA: Improving Non-autoregressive Machine Translation with Dependency-Aware Decoder](https://arxiv.org/abs/2203.16266)  |  ACL 2023  | [![Github](https://img.shields.io/github/stars/zhanjiaao/NAT_DePA?style=flat)](https://github.com/zhanjiaao/NAT_DePA) | 
| [Directed acyclic transformer for non-autoregressive machine translation](https://proceedings.mlr.press/v162/huang22m)  |  ICML 2022  | [![Github](https://img.shields.io/github/stars/thu-coai/DA-Transformer?style=flat)](https://github.com/thu-coai/DA-Transformer) | 
| [Viterbi decoding of directed acyclic transformer for non-autoregressive machine translation](https://arxiv.org/abs/2210.05193)  |  EMNLP 2022  | [![Github](https://img.shields.io/github/stars/thu-coai/DA-Transformer?style=flat)](https://github.com/thu-coai/DA-Transformer) | 
| [Fully Non-autoregressive Neural Machine Translation: Tricks of the Trade](https://arxiv.org/abs/2012.15833)  |  ACL-IJCNLP 2021  | - | 
| [Aligned cross entropy for non-autoregressive machine translation](https://proceedings.mlr.press/v119/ghazvininejad20a.html)  |  ICML 2020  | [![Github](https://img.shields.io/github/stars/m3yrin/aligned-cross-entropy?style=flat)](https://github.com/m3yrin/aligned-cross-entropy) | 
| [ngram-OAXE: Phrase-based order-agnostic cross entropy for non-autoregressive machine translation](https://arxiv.org/abs/2210.03999)  |  COLING 2022  | [![Github](https://img.shields.io/github/stars/ltencent-ailab/machine-translation?style=flat)](https://github.com/tencent-ailab/machine-translation/COLING22_ngram-OAXE) | 
| [Multi-granularity optimization for non-autoregressive translation](https://arxiv.org/abs/2210.11017)  |  EMNLP 2022  | [![Github](https://img.shields.io/github/stars/yafuly/MGMO-NAT?style=flat)](https://github.com/yafuly/MGMO-NAT) | 
| [One reference is not enough: Diverse distillation with reference selection for non-autoregressive translation](https://arxiv.org/abs/2205.14333)  |  NAACL 2022  | [![Github](https://img.shields.io/github/stars/ictnlp/DDRS-NAT?style=flat)](https://github.com/ictnlp/DDRS-NAT) | 

#### Masked Generation

| Paper      | Venue       | Code   |  
|-----------|:----------------:|:--------------:| 
| [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)  |  arxiv 2023  | [![Github](https://img.shields.io/github/stars/shreyansh26/Speculative-Sampling?style=flat)](https://github.com/shreyansh26/Speculative-Sampling) | 
| [Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling](https://arxiv.org/abs/2409.02908)  |  ICLR 2025  | - | 
| [A continuous time framework for discrete denoising models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html)  |  NeurIPS 2022  | [![Github](https://img.shields.io/github/stars/andrew-cr/tauLDR?style=flat)](https://github.com/andrew-cr/tauLDR) | 
| [Discrete diffusion modeling by estimating the ratios of the data distribution](https://arxiv.org/abs/2310.16834)  |  ICML 2024  | [![Github](https://img.shields.io/github/stars/louaaron/Score-Entropy-Discrete-Diffusion?style=flat)](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) | 
| [Simplified and generalized masked diffusion for discrete data](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html)  |  NeurIPS 2024  | [![Github](https://img.shields.io/github/stars/darioShar/pytorch-md4?style=flat)](https://github.com/darioShar/pytorch-md4) | 
| [Seed Diffusion](https://arxiv.org/abs/2508.02193)  |  arxiv 2025  | - | 
| [Target concrete score matching: A holistic framework for discrete diffusion](https://arxiv.org/abs/2504.16431)  |  ICML 2025  | - | 
| [Discrete diffusion modeling by estimating the ratios of the data distribution](https://arxiv.org/abs/2310.16834)  |  ICML 2024  | [![Github](https://img.shields.io/github/stars/louaaron/Score-Entropy-Discrete-Diffusion?style=flat)](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) | 
| [Score-based continuous-time discrete diffusion models](https://arxiv.org/abs/2211.16750)  |  ICLR 2023  | - | 
| [Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding](https://arxiv.org/abs/2505.22618)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/NVlabs/Fast-dLLM?style=flat)](https://github.com/NVlabs/Fast-dLLM) | 
| [Large language diffusion models](https://arxiv.org/abs/2502.09992)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/ML-GSAI/LLaDA?style=flat)](https://github.com/ML-GSAI/LLaDA) | 
| [Beyond autoregression: Discrete diffusion for complex reasoning and planning](https://arxiv.org/abs/2410.14157)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/HKUNLP/diffusion-vs-ar?style=flat)](https://github.com/HKUNLP/diffusion-vs-ar) | 
| [A reparameterized discrete diffusion model for text generation](https://arxiv.org/abs/2302.05737)  |  COLM 2024  | [![Github](https://img.shields.io/github/stars/hkunlp/reparam-discrete-diffusion?style=flat)](https://github.com/hkunlp/reparam-discrete-diffusion) | 
| [Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions](https://arxiv.org/abs/2502.06768)  |  ICML 2025  | - | 
| [Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking](https://arxiv.org/abs/2505.24857)  |  arxiv 2025  | - | 
| [Accelerating Diffusion Large Language Models with SlowFast: The Three Golden Principles](https://arxiv.org/abs/2506.10848)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/LiangrunFlora/Slow-Fast-Sampling?style=flat)](https://github.com/LiangrunFlora/Slow-Fast-Sampling) | 
| [A continuous time framework for discrete denoising models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html)  |  NeurIPS 2022  | [![Github](https://img.shields.io/github/stars/andrew-cr/tauLDR?style=flat)](https://github.com/andrew-cr/tauLDR) | 
| [Remasking discrete diffusion models with inference-time scaling](https://arxiv.org/abs/2503.00307)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/kuleshov-group/remdm?style=flat)](https://github.com/kuleshov-group/remdm) | 
| [Simplified and generalized masked diffusion for discrete data](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html)  |  NeurIPS 2024  | [![Github](https://img.shields.io/github/stars/google-deepmind/md4?style=flat)](https://github.com/google-deepmind/md4) | 
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
| [Dream 7B](https://arxiv.org/abs/2508.15487)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/DreamLM/Dream?style=flat)](https://github.com/DreamLM/Dream) | 
| [DIFFPO: Diffusion-styled Preference Optimization for Efficient Inference-Time Alignment of Large Language Models](https://arxiv.org/abs/2503.04240)  |  ACL 2025  | [![Github](https://img.shields.io/github/stars/zjuruizhechen/DiffPO?style=flat)](https://github.com/zjuruizhechen/DiffPO) | 
| [Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs](https://arxiv.org/abs/2507.18578)  |  arxiv 2025  | [![Github](https://img.shields.io/github/stars/Feng-Hong/WINO-DLLM?style=flat)](https://github.com/Feng-Hong/WINO-DLLM) |

#### Edit-Based Refinement

| Paper      | Venue       | Code   |  
|-----------|:----------------:|:--------------:| 
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
| [RenewNAT: renewing potential translation for non-autoregressive transformer](https://ojs.aaai.org/index.php/AAAI/article/view/34647)  |  AAAI 2023  | - | 
| [Learning to recover from multi-modality errors for non-autoregressive neural machine translation](https://arxiv.org/abs/2006.05165)  |  ACL 2020  | [![Github](https://img.shields.io/github/stars/ranqiu92/RecoverSAT?style=flat)](https://github.com/ranqiu92/RecoverSAT) | 
| [Hybrid-regressive neural machine translation](https://arxiv.org/abs/2210.10416)  |  ICLR 2023  | - | 
| [Iterative Translation Refinement with Large Language Models](https://arxiv.org/abs/2306.03856)  |  EAMT 2024  | - | 
| [IterGen: Iterative Semantic-aware Structured LLM Generation with Backtracking](https://arxiv.org/abs/2410.07295)  |  ICLR 2025  | [![Github](https://img.shields.io/github/stars/structuredllm/itergen?style=flat)](https://github.com/structuredllm/itergen) | 
| [Rejuvenating low-frequency words: Making the most of parallel data in non-autoregressive translation](https://arxiv.org/abs/2106.00903)  |  ACL 2021  | [![Github](https://img.shields.io/github/stars/alphadl/RLFW-NAT?style=flat)](https://github.com/alphadl/RLFW-NAT) | 
| [Understanding and Improving Lexical Choice in Non-Autoregressive Translation](https://arxiv.org/abs/2012.14583)  |  ICLR 2021  | [![Github](https://img.shields.io/github/stars/alphadl/LCNAT?style=flat)](https://github.com/alphadl/LCNAT) | 
| [SlotRefine: A fast non-autoregressive model for joint intent detection and slot filling](https://arxiv.org/abs/2010.02693)  |  EMNLP 2020  | [![Github](https://img.shields.io/github/stars/moore3930/SlotRefine?style=flat)](https://github.com/moore3930/SlotRefine) | 
| [Non-autoregressive dialog state tracking](https://arxiv.org/abs/2002.08024)  |  ICLR 2020  | [![Github](https://img.shields.io/github/stars/henryhungle/NADST?style=flat)](https://github.com/henryhungle/NADST) | 
