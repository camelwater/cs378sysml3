# CS 378 Serving Assignment (Vidur)

A slim [Vidur](https://github.com/microsoft/vidur) fork for the language-model
serving assignment. See **`assignment.md`** for the full handout: setup, every
part, questions, submission, and grading.

The assignment follows one hour of real chat traffic
(`traces/azure_conversation.csv`) on Llama-3-8B and one A100-80GB GPU:
estimate where one GPU falls behind (Part 1), check it in the simulator
(Part 2), fix stalled streaming with a per-step token budget (Part 3), decide
who waits (Part 4), then add replicas and route between them (Part 5).

## No GPU needed: everything is simulated on your CPU

There is **no real A100** anywhere in this assignment, and you do not need
CUDA, PyTorch, or a GPU.

- Vidur is a discrete-event simulator written in plain Python. It never loads
  model weights, runs a kernel, or allocates GPU memory. It tracks which
  requests are queued and running and how many KV-cache blocks each holds,
  and it advances a simulated clock.
- How long each simulated step takes is predicted by a regression model fit
  to kernel timings that the Vidur authors profiled on real A100s
  (`data/profiling/compute/a100/`). Your laptop's speed changes how long a
  run takes, not the results it reports.
- All waits, time-to-first-token, and time-between-tokens numbers are in
  simulated seconds. One run replays an hour of trace in a few minutes.
- "A100-80GB" means the simulated device: 80 GiB of memory and A100 step
  times.

## Setup

Python **3.10** . From this directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Files you edit

| File | Part |
|---|---|
| `vidur/scheduler/utils/memory_planner.py` | Warmup — the full `MemoryPlanner` |
| `analysis/workload.py` | Part 1 — arrival rates, queue wait, KV cache in use |
| `vidur/scheduler/replica_scheduler/sarathi_replica_scheduler.py` | Part 3 — tokens per request and the four batch-building steps |
| `vidur/scheduler/replica_scheduler/spf_replica_scheduler.py` | Part 4 — priority with aging, and queue order |
| `vidur/scheduler/global_scheduler/lor_global_scheduler.py` | Part 5 — least outstanding requests router |
| `vidur/scheduler/global_scheduler/lot_global_scheduler.py` | Part 5 — least outstanding tokens router |

Fill only the `# [Warmup] TODO` / `# [Part N] TODO` blocks. Notebooks for each
part are in `notebooks/`.

## Repository map

| Path | What it is |
|---|---|
| `assignment.md` | The handout |
| `analysis/` | Part 1 estimates (`workload.py`) and the helpers notebooks use to read Vidur runs (`replay.py`) |
| `notebooks/` | One notebook per part: `workload`, `replay`, `chunking`, `spf`, `replicas` |
| `scripts/run_default.sh` | Runs one Vidur replay with the assignment's settings |
| `scripts/run_tests.sh` | Runs every test with pytest |
| `tests/` | Autograded tests, one file per coding part |
| `traces/` | `azure_conversation.csv` (real, 19,366 requests) and `long_short_bursts.csv` (synthetic, for routers) |
| `data/profiling/` | Kernel timings measured on real GPUs; Vidur's step-time model is fit to these |
| `vidur/` | The simulator |

## Running

```bash
bash scripts/run_tests.sh                       # every test (pytest)
QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv               # vLLM
SCHED=sarathi CHUNK=512 QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
SCHED=spf CHUNK=512 AGING=100 QPS=5.53 bash scripts/run_default.sh traces/azure_conversation.csv
REPLICAS=2 GLOBAL=lot SCHED=sarathi CHUNK=512 QPS=8 bash scripts/run_default.sh traces/long_short_bursts.csv
```

Each replay of `azure_conversation.csv` takes about 3 minutes and a few GB of
RAM (two replicas: about 5 minutes and more memory); run them one at a
time. `traces/long_short_bursts.csv` replays in seconds.

The wrappers always use `--execution_time_predictor_config_type
linear_regression`. Vidur's default random-forest predictor can spend tens of
minutes training on a laptop before it produces anything.

## Output

Each run writes `simulator_output/<timestamp>/`. The notebooks read these
files through `analysis/replay.py`:

| File | Contents |
|---|---|
| `config.json` | The run's full configuration |
| `request_metrics.csv` | One row per request: wait, time to first token, end-to-end time |
| `plots/decode_token_execution_plus_preemption_time.csv` | CDF of time between tokens |
| `replica_<i>_kv_usage.csv` | KV-cache blocks in use over time, per replica |
| `plots/replica_<i>_stage_1_busy_time_percent.json` | Share of the run each replica was busy |

Replica files are numbered from 1: `replica_1_*` is replica id 0 in the
routers, and `replica_2_*` is replica id 1.

## Acknowledgments

Simulator code is from Microsoft / Georgia Tech [Vidur](https://github.com/microsoft/vidur)
(MLSys 2024). Course scripts and TODOs are CS 378 only.
