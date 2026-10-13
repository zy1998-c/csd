# CSD on RWKU

Counterfactual Self-Distillation (CSD) 的训练和 RWKU 测试流程。保留当前模型生成响应、在反事实上下文中计算无梯度教师分布、对生成 token 最小化 `KL(teacher || student)` 的算法。训练不使用 retain loss，也不使用 RWKU 测试答案。

## 1. 创建环境

推荐 Linux、Python 3.10、NVIDIA GPU 和 CUDA 12.1 兼容驱动。8B LoRA 建议 40GB 以上显存，实际用量取决于上下文、K 和生成长度；8B 全参数 AdamW 的模型、梯度和优化器需要约 128GB 加激活显存，应使用足够显存或自行配置 ZeRO。先用下面的小模型测试确认环境。

```bash
git clone https://github.com/zy1998-c/csd.git
cd csd
conda create -n csd-rwku python=3.10 -y
conda activate csd-rwku
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
bash scripts/setup.sh
python -m pip check
python scripts/smoke_test.py
```

没有 conda 时可使用 `python3.10 -m venv .venv` 和 `source .venv/bin/activate`。CPU/Mac 安装时跳过 CUDA wheel 命令。`setup.sh` 安装固定版本依赖、下载 NLTK punkt，并在 `vendor/RWKU` 下载固定的官方 RWKU 底座：`b8a03b3ce34fb4a96001df545a56558d75a078a3`。不要使用最新版 LLaMA-Factory，它已从 `llmtuner` 改为 `llamafactory`，且缺少 RWKU 模块。

`smoke_test.py` 不下载模型或 RWKU 数据，使用本地随机小模型和测试样本，实际执行全参数与 LoRA 的训练、验证、保存、重载及全部九组评估。它只验证工程流程，不代表遗忘效果。

## 2. 下载 RWKU 正式测试数据

```bash
python scripts/prepare_rwku.py --check-only
python scripts/prepare_rwku.py
```

已有十个目标的 `reject_replace01.json` 是 CSD 训练集，无需再次生成。下载脚本按其中的 `subject` 从 [官方 Hugging Face 数据集](https://huggingface.co/datasets/jinzhuoran/RWKU) 筛选并保存全部 12 个测试子集。`data/rwku_manifest.json` 记录下载的 commit 和各目标样本数；需要复现时添加 `--revision <该commit>`。只下载一个目标可添加 `--targets 1_Stephen_King`。数据不足或为空会报错。

当前只提供十个目标的反事实训练数据；200 个目标的扩展需要先提供相应训练数据和 registry，不能把测试集直接转换成训练集。

## 3. 选择模型

正式实验可使用 `meta-llama/Meta-Llama-3-8B-Instruct` + `TEMPLATE=llama3`，与 RWKU 原项目相同。Llama-3.1 也可使用 `llama3` 模板。Meta 模型需要先在 Hugging Face 接受模型许可，再运行：

```bash
huggingface-cli login
export MODEL_NAME_OR_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export TEMPLATE=llama3
```

无需申请访问权限的小模型：`HuggingFaceTB/SmolLM2-135M-Instruct` + `TEMPLATE=chatml`。它适合验证流程，不能替代论文的 8B 结果。

## 4. 训练

LoRA，单目标：

```bash
export DATASET=1_Stephen_King_Reject_Replace01
export OUTPUT_DIR="$PWD/outputs/lora/$DATASET"
bash scripts/run_csd_lora.sh
```

全参数训练：

```bash
OUTPUT_DIR="$PWD/outputs/full/$DATASET" bash scripts/run_csd_full.sh
```

脚本默认 BF16、batch size 1、K=3、cutoff 512、rollout 64 token、3 epochs。旧 GPU 可设置 `PRECISION_FLAG=--fp16`。FP32/CPU 使用 `PRECISION_FLAG=''`，并在脚本末尾传入 `--use_cpu`。LoRA 的批大小可用 `BATCH_SIZE` 调整，其余 Trainer 参数直接附加在命令末尾。

独立的训练验证集通过 `--val_size` 从 CSD 训练数据中划分，验证的是 CSD KL loss；正式 RWKU 测试由下一节执行：

```bash
bash scripts/run_csd_lora.sh --do_eval --val_size 0.1
# 中断后恢复，不要复用一个已有实验目录开启新实验：
bash scripts/run_csd_lora.sh --resume_from_checkpoint "$OUTPUT_DIR/checkpoint-500"
```

在 CPU 上运行真实小模型的快速检查（仍需下载模型）：

```bash
MODEL_NAME_OR_PATH=HuggingFaceTB/SmolLM2-135M-Instruct \
TEMPLATE=chatml PRECISION_FLAG='' OUTPUT_DIR="$PWD/outputs/small-lora" \
bash scripts/run_csd_lora.sh --use_cpu --max_steps 2 --max_samples 8 \
  --cutoff_len 128 --csd_rollout_max_new_tokens 8 \
  --gradient_accumulation_steps 1 --do_eval --val_size 0.25 --report_to none
```

## 5. 正式测试与前后比较

原始模型基线：

```bash
python scripts/evaluate_rwku.py --model "$MODEL_NAME_OR_PATH" \
  --target 1_Stephen_King --output-dir results/1_Stephen_King/base
```

LoRA 模型，需要同时提供原始底座和训练得到的 adapter：

```bash
python scripts/evaluate_rwku.py --model "$MODEL_NAME_OR_PATH" \
  --adapter "$OUTPUT_DIR" --target 1_Stephen_King \
  --output-dir results/1_Stephen_King/csd-lora
```

全参数模型直接提供完整检查点，不传 adapter：

```bash
python scripts/evaluate_rwku.py \
  --model "outputs/full/1_Stephen_King_Reject_Replace01" \
  --target 1_Stephen_King --output-dir results/1_Stephen_King/csd-full
```

默认运行全部九组：forget、neighbor、MMLU、BBH、TruthfulQA、TriviaQA、forget MIA、retain MIA、fluency。默认 batch size 1 避免长上下文评估显存不足；可加 `--batch-size 4`。CPU 加 `--device cpu --dtype float32`，旧 GPU 加 `--dtype float16`。

每组输出逐样本结果，`summary.json` 汇总指标和数据来源。Forget 的 ROUGE-L recall 越低通常表示遗忘越强，neighbor 与 utility 应与基线比较以观察能力损失。MIA 输出官方 loss/zlib/min-k/min-k++ 分数，不将这些原始分数误称为攻击 AUC。

仅快速检查时可加 `--max-samples 1` 或 `--suites forget neighbor`。前者会将结果标记为 `smoke_test`，不能用作正式成绩。测试目录必须为空，防止覆盖已有结果。生成报错会终止评估，不能用空答案掩盖 OOM。

## 6. 批量运行十个目标

每个目标从相同的原始模型独立训练，避免把前一个目标的 adapter 当作下一个的底座：

```bash
for path in data/RWKU/Target/*/reject_replace01.json; do
  target=$(basename "$(dirname "$path")")
  dataset="${target}_Reject_Replace01"
  out="$PWD/outputs/lora/$dataset"
  DATASET="$dataset" OUTPUT_DIR="$out" bash scripts/run_csd_lora.sh
  python scripts/evaluate_rwku.py --model "$MODEL_NAME_OR_PATH" \
    --target "$target" --output-dir "results/$target/base"
  python scripts/evaluate_rwku.py --model "$MODEL_NAME_OR_PATH" \
    --adapter "$out" --target "$target" --output-dir "results/$target/csd"
done
```

## 实现与来源

- `overlay/src/llmtuner/train/csd/`：原 CSD 算法，修复 EOS 后填充参与 KL、混合精度 KL 数值稳定性、左填充 position IDs 和验证入口。
- `scripts/setup.sh`：固定 RWKU 底座和依赖，自动应用 overlay。
- `scripts/prepare_rwku.py`：独立测试集准备和训练数据校验。
- `scripts/evaluate_rwku.py`：模型/adapter 加载和官方 RWKU 全套评估。
- `overlay/src/llmtuner/eval/`：来自官方 RWKU 的评估代码，修复错误吞掉、设备放置、重复 BOS、TruthfulQA 选项缺失/概率下溢和短文本 MIA 数值问题。

参考 [RWKU](https://github.com/jinzhuoran/RWKU)、[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。模型和数据分别遵循各自许可，使用上游代码时请核查原项目许可。已执行的验证和未验证的硬件路径见 [VALIDATION.md](VALIDATION.md)。
