# Nemotron: using and training

Setup state on this machine: core package + `wandb` and `data-sdg` extras are
installed in `.venv-core`. There is no GPU here, so inference servers and real
training must run on a GPU machine or Slurm cluster.

```bash
source .venv-core/bin/activate
nemotron --help
nemotron steps list
nemotron steps run sft/automodel -c tiny --dry-run   # inspect config, no execution
```

## A. Using the models

Model weights: https://huggingface.co/collections/nvidia/nvidia-nemotron-v3
Deployment guides (TensorRT-LLM, vLLM, SGLang, NIM, HF):
https://docs.nvidia.com/nemotron/latest/deployment-guides.html

**Option 1: hosted API (no GPU needed).** Get a key from build.nvidia.com,
pick a Nemotron model there, copy its model id, then:

```bash
export NEMOTRON_API_KEY="nvapi-..."
export NEMOTRON_MODEL="<model id from the model page>"
python chat.py "Hello"
```

**Option 2: self-host with vLLM (needs a GPU big enough for the model).**
Follow the vLLM page in the deployment guides for the exact command and
flags for the model you pick, then point `NEMOTRON_BASE_URL` at
`http://localhost:8000/v1` and run `chat.py` the same way.

Rough sizing (total parameters must fit in GPU memory even though fewer are active):
Lightning 3.5 (30B) and Nano 3 (31.6B) fit on one large GPU when quantized or
on a couple of GPUs in bf16; Super (120B) and Ultra (550B) need multi-GPU nodes.

## B. Training / customizing

Estimate GPUs, time and cost first:
https://docs.nvidia.com/nemotron/latest/train-models/sizing-calculator.html

On a GPU machine or Slurm login node:

```bash
git clone https://github.com/NVIDIA-NeMo/Nemotron
cd Nemotron && uv sync
```

Set up an executor profile for your cluster (see "Execution through NeMo-Run"
and "Environment Profiles and Executors" in the docs), then:

```bash
# smoke test: tiny SFT job
uv run nemotron steps run sft/automodel -c tiny --run YOUR-CLUSTER

# SFT on your own data
#   docs: train-models/how-to/run-sft-automodel.html

# full Nano3 pipeline, stage by stage
uv run nemotron nano3 data prep sft --run YOUR-CLUSTER
uv run nemotron nano3 sft --run YOUR-CLUSTER
uv run nemotron nano3 data prep rl --run YOUR-CLUSTER
uv run nemotron nano3 rl --run YOUR-CLUSTER
```

Customization workflows that also run as steps: data curation (`curate/nemo_curator`),
synthetic data (`sdg/data_designer`), translation, MCQ benchmarks (`byob/mcq`),
evaluation (`eval/model_eval`), plus PEFT/LoRA, DPO/RLVR/RLHF, quantize/prune/distill.

Use `--dry-run` on any command to print the compiled config before spending GPU time.
