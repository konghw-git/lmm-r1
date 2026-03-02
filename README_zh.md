<div align="center">
    <img alt="LMM-R1 logo" src="output/lmm-r1-logo-panda.png" style="height: 140px;" />
</div>

# LMM-R1: 通过两阶段规则型强化学习增强3B大型多模态模型的推理能力

<div align="center">
<p align="center">
      <a href="https://github.com/TideDra/lmm-r1/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/TideDra/lmm-r1" />
      </a>
      <a href="https://github.com/TideDra/lmm-r1/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/TideDra/lmm-r1?color=0088ff" />
      </a>
      <a href="https://github.com/TideDra/lmm-r1/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/TideDra/lmm-r1?color=0088ff" />
      </a>
      <a href="https://github.com/TideDra/lmm-r1/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/TideDra/lmm-r1?color=0088ff" />
      <a href="https://github.com/TideDra/lmm-r1/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/TideDra/lmm-r1?color=ccf" />
      </a>
      <br>
      <em>开源 / 全面 / 轻量 / 易用</em>
    </p>
</p>
</div>

<hr>

[![🤗 HF 数据集](https://img.shields.io/badge/🤗-数据集-yellow)](https://huggingface.co/datasets/VLM-Reasoner/VerMulti) [![🤗 HF 模型](https://img.shields.io/badge/🤗-模型-blue)](https://huggingface.co/VLM-Reasoner/LMM-R1-MGT-PerceReason) [![📄 论文](https://img.shields.io/badge/📄-论文-green)](https://arxiv.org/pdf/2503.07536) [![🌐 项目主页](https://img.shields.io/badge/🌐-项目主页-purple)](https://tidedra.github.io/lmm-r1-project/)

[切换到英文版 (Switch to English version)](/README.md)

## 新闻
- [2025/3/11] 🚀 我们的代码被合并进了[OpenRLHF-M](https://github.com/OpenRLHF/OpenRLHF-M), 由OpenRLHF官方开发的多模态强化学习框架。
- [2025/3/11] ✨ 我们发布了论文 "[LMM-R1: 通过两阶段规则型强化学习增强3B大型多模态模型的推理能力](https://arxiv.org/pdf/2503.07536)"！

- [2025/2/13] 我们发布了LMM-R1的代码！

## 简介

小型3B参数量的大型多模态模型(LMMs)在推理任务上面临挑战，这主要是由于其有限的参数容量以及视觉感知与逻辑推理整合的内在复杂性。高质量的多模态推理数据也非常稀缺，进一步增加了训练难度。为了解决这些挑战，我们提出了**LMM-R1**，一个两阶段规则型强化学习框架，能够高效地增强推理能力：

1. **基础推理增强(FRE)**：使用纯文本数据建立强大的推理基础
2. **多模态泛化训练(MGT)**：将这些能力扩展到多模态领域

这种方法克服了数据限制，同时显著提高了模型在各种推理任务上的表现。

![pipeline](output/model.jpg)

## 例子
**几何题目:**

![motivation](output/motivation.png)

**推箱子:**

![sobokan_deom](output/sokoban_demo.gif)
 
## 快速开始

### 安装

```bash
git clone https://github.com/TideDra/lmm-r1.git
cd lmm-r1
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
```

> [!注意]
>我们推荐使用vLLM 0.7.2或更高版本。
>我们还提供了[vLLM的Docker文件](./dockerfile/)和[Nvidia-Docker一键安装脚本](./examples/scripts/nvidia_docker_install.sh)。

### 准备数据集

LMM-R1要求多模态提示数据集采用OpenAI兼容的消息格式：
```json
[
  {
    "message":"[
      {
        \"role\": \"user\",
        \"content\": [
            { \
                \"type\": \"image\",
                \"image\": \"file:///path/to/your/image.jpg\",
            }, \
            {\"type\": \"text\", \"text\": \"图片中有多少只猫？\"},
        ],
      }
    ]",
    "answer": "$3$"
  },
]
```
**注意message是一个字符串化的列表。**
参考示例数据集`examples/data/test_message.jsonl`。

- 我们可以使用`--input_key`指定输入数据集的`JSON键名`，如`--prompt_data {name or path}`(PPO)或`--dataset {name or path}`。**不要**对多模态提示使用`--apply_chat_template`，消息将在内部处理。
- OpenRLHF还支持使用`--prompt_data_probs 0.1,0.4,0.5`(PPO)或`--dataset_probs 0.1,0.4,0.5`混合多个数据集。

### 训练

我们的训练过程遵循论文中描述的两阶段方法。我们为每个阶段提供脚本，以便复现我们的结果。

#### 阶段1：基础推理增强(FRE)

这个阶段专注于使用纯文本数据增强模型的推理能力。

```bash
# 使用纯文本数据训练(FRE-Text)
bash examples/scripts/lmm_r1/train_grpo_text.sh

# 使用多模态数据训练(FRE-Multi)作为比较
bash examples/scripts/lmm_r1/train_fre_multi.sh
```

FRE-Text脚本使用[DeepScaler-40K](https://huggingface.co/datasets/VLM-Reasoner/deepscaler)数据集通过规则型强化学习增强模型的基础推理能力。这个阶段对于在进入多模态任务前建立强大的推理能力至关重要。

#### 阶段2：多模态泛化训练(MGT)

这个阶段通过在特定任务上继续训练，将推理能力扩展到多模态领域。

```bash
# 在几何领域训练(MGT-Geo)
bash examples/scripts/lmm_r1/train_mgt_geo.sh

# 在感知-推理平衡领域训练(MGT-PerceReason)
bash examples/scripts/lmm_r1/train_mgt_percereas.sh
```

每个MGT脚本都从FRE-Text检查点继续训练，专注于特定领域：
- **MGT-Geo**：使用[VerMulti-Geo]((https://huggingface.co/datasets/VLM-Reasoner/VerMulti))数据集(15K几何问题)增强几何推理
- **MGT-PerceReason**：使用完整的[VerMulti](https://huggingface.co/datasets/VLM-Reasoner/VerMulti)数据集平衡感知和推理能力

我们开源了最终模型 [MGT-PerceReason](https://huggingface.co/VLM-Reasoner/LMM-R1-MGT-PerceReason)

#### 直接强化学习训练(用于比较)

我们还提供了不经过FRE阶段的直接强化学习训练脚本，这些脚本在我们的论文中用作比较基准：

```bash
# 几何领域的直接强化学习训练
bash examples/scripts/lmm_r1/train_direct_rl_geo.sh
```

这些脚本直接在特定领域数据上训练基线模型，跳过FRE阶段，这有助于展示我们两阶段方法的有效性。

## 功能特点

LMM-R1是[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)的一个分支，旨在提供高性能的LMM强化学习基础设施，以增强多模态推理能力。我们目前支持LMM的PPO/REINFORCE++/RLOO训练，并且与[R1-V](https://github.com/Deep-Agent/R1-V)(GRPO)相比，实现了4.7倍的加速(RLOO)。

![time_compare](output/time_compare.jpg)

- 支持LMM训练(Qwen2-VL, Qwen2.5-VL)
- 基于Ray的分布式[PPO](./examples/scripts/train_ppo_llama_ray.sh)和[REINFORCE++/RLOO](./examples/scripts/train_reinforce_llama_ray.sh)实现
- [基于Ray的强化微调](./examples/scripts/train_ppo_llama_with_reward_fn.sh)
- 支持使用混合引擎的基于Ray的[PPO](./examples/scripts/train_ppo_llama_ray_hybrid_engine.sh)和[REINFORCE++/RLOO](./examples/scripts/train_reinforce_llama_ray_hybrid_engine.sh)(`--colocate_all_models`, `--vllm_enable_sleep`和`--vllm_gpu_memory_utilization 0.5`)
- 完全支持[超过700亿参数模型](./examples/scripts/train_ppo_llama_ray_70b.sh)的RLHF微调
- 集成vLLM以加速RLHF任务中的生成(`--vllm_num_engines`)
- 支持多个奖励模型(`--reward_pretrain model1,model2...`)和远程奖励模型(`--remote_rm_url`)
- 集成FlashAttention2(`--flash_attn`)
- 支持QLoRA(`--load_in_4bit`)和[LoRA](./examples/scripts/train_sft_mixtral_lora.sh)(`--lora_rank`, `--target_modules`)
- 兼容HuggingFace的`tokenizer.apply_chat_template`用于数据集(`--apply_chat_template`和`--input_key`)
- 支持Wandb(`--use_wandb`)和TensorBoard(`--use_tensorboard`)日志记录
- 检查点恢复功能(`--load_checkpoint`和`--save_steps`)
- 提供多节点训练脚本，如[Ray PPO](./examples/scripts/train_ppo_llama_ray_slurm.sh)

## 参考文献与致谢
我们衷心感谢[DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1)在LLM推理方面的探索，以及[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)提供的出色RL基础设施。我们还要感谢[open-r1](https://github.com/huggingface/open-r1)和[simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)，它们为我们复现R1提供了见解。特别感谢[杨凯](https://github.com/yangkai798)、[刘杰](https://jieliu.site/)、[游志远](https://zhiyuanyou.github.io/)提供的宝贵建议。

- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) 
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [open-r1](https://github.com/huggingface/open-r1)
- [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)

## 引用
如果您发现LMM-R1对您的研究和应用有用，请使用以下BibTeX进行引用：

```bib
@article{peng2025lmmr1,
  title={LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL},
  author={Peng, Yingzhe and Zhang, Gongrui and Zhang, Miaosen and You, Zhiyuan and Liu, Jie and Zhu, Qipeng and Yang, Kai and Xu, Xingzhong and Geng, Xin and Yang, Xu},
  journal={arXiv preprint arXiv:2503.07536},
  year={2025}
}
```