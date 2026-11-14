# Familiar or capable?

Does renaming tools to unfamiliar but equally clear names change multi-turn tool-calling
results? A paired pilot on BFCL `multi_turn_base`. Plan: [`plan_pilot.md`](plan_pilot.md).

**Status:** steps 1–2 done (pipeline built and tested locally; 20 tasks drawn and frozen).
No model results yet.

## Pre-registration

Fixed before any model run. Nothing here changes after results are seen; if something must
change (a bug), the change and the reason go below with a date.

**Hypotheses**

- **H1 (gap):** task success is lower on the renamed twin than on the original, for the same task and model.
- **H2 (where):** the loss shows up mainly in tool selection and in later turns, less in argument filling.
- **H3 (intrusion):** on the twin, models sometimes call the *original* tool name (for example `ls`) even though it is not in the tool list.
- **H0:** no meaningful difference. A valid outcome, reported as such.

**Decision rule.** Per model, over the 20 paired tasks: b = pass original / fail twin,
c = fail original / pass twin, net = b − c (one task = 5 points), p = exact one-sided McNemar
on (b, c). Intrusions = twin-condition calls to an original *function* name not in the tool
list (original *argument* names are counted separately and are not part of H3). "Models" =
those passing every validity gate in plan 8.2, including Floor. Rows are checked top to
bottom; the first match is the outcome. Implemented in `src/decide.py`.

| # | Outcome | Condition |
|---|---|---|
| 1 | Strong signal | p < 0.05 in ≥ 1 model **and** net ≥ 3 in a second model |
| 2 | Suggestive | net ≥ 3 in ≥ 2 models |
| 3 | Reversed | net ≤ −3 in ≥ 2 models |
| 4 | Behaviour without score change | net ≤ 2 in every model, and H3 holds in ≥ 1 model |
| 5 | Null | \|net\| ≤ 1 in every model, and H3 fails in every model |
| 6 | Inconclusive | anything else |

H3 holds for a model if it makes ≥ 3 intrusions across ≥ 2 tasks, more than its calls to
non-existent tools in the original condition. H2 is assessed only with ≥ 5 original-pass /
twin-fail tasks pooled across models, and holds if ≥ 60% of them fail first on tool choice
(wrong tool or intrusion) rather than arguments. Wording allowed per outcome: plan section 2.

**Tasks.** 20 of 200 `multi_turn_base` tasks, drawn by `scripts/select_tasks.py` with seed
20261114, stratified by main backend: 8 file system, 4 vehicle control, 4 trading, 4 travel.
IDs in `tasks/selected_ids.json`; strata and reserve order in `tasks/selection.json`. No
outcome-based exclusions. A reserve replaces a task only for a pipeline reason found before
any model run, recorded in `tasks/selection.json`. All 20 pass the ground-truth replay in the
original condition.

Still to freeze before the first real run (plan 8.1): name mappings and clarity check (steps
3–4), model revisions, vLLM version and sampling settings.

**Changes after freezing:** none.

## Setup

```sh
uv sync                 # Python 3.11, pinned deps in uv.lock
uv run pytest           # 62 tests, no GPU or server needed
```

Local smoke run (Apple Silicon; our own raw server, see below):

```sh
uv run --no-project --with mlx-lm==0.32.2 python scripts/mlx_raw_server.py \
    --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
uv run python -m src.run --model qwen3-local --condition original --ids tasks/dev_ids.json \
    --endpoint http://127.0.0.1:8080 --server-model Qwen/Qwen3-1.7B \
    --server "mlx_raw_server(mlx-lm==0.32.2)" --run-id local1
```

Inference servers stay outside the project environment (vLLM and mlx-lm pin their own
stacks; bfcl-eval pins `numpy==1.26.4`). The client talks to them only over HTTP.

## Pipeline

What we use from BFCL (`bfcl-eval==2026.3.23`): task data, backend classes,
`execute_multi_turn_func_call`, `multi_turn_checker`. What we don't: its model handlers.
Prompts are rendered with each model's own HF chat template (`src/render.py`), sent as raw
text to `/v1/completions` (`src/client.py`), parsed by our code (`src/parsers/`), checked
against the tool list (`src/mapping.py`), executed on BFCL's backends, and scored by BFCL's
checker (`src/score.py`). The loop (`src/loop.py`) mirrors BFCL's `inference_multi_turn_FC`:
the turn ends on a call-free reply or a parse failure; more than 20 steps in a turn fails the entry.

Every step is logged to `results/raw/<run>.jsonl`: prompt, request settings, raw output,
finish reason, parse status, parsed/mapped/rejected calls, tool outputs; plus one record per
task with the checker verdict.

## Versions

| | |
|---|---|
| bfcl-eval | 2026.3.23 |
| transformers | 5.23.0 |
| Qwen/Qwen3-1.7B (local test only) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| allenai/Olmo-3-7B-Instruct (template read) | 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc |
| llm-jp/llm-jp-4.1-8b-thinking (template read) | 7c19bc90e49b717381e4f474ee95964c05217a28 |

Model revisions for the real run are pinned in `src/models.py` before it starts.

## Notes and deviations (so far)

- **mlx_lm.server is not a raw server.** In mlx-lm 0.32.2 it strips tool-call and reasoning
  text from `/v1/completions` output and reports `finish_reason: "tool_calls"`. We use our own
  small server (`scripts/mlx_raw_server.py`) for local tests, and the client now refuses any
  finish reason other than `stop`/`length`.
- **Greedy decoding** (`temperature=0`). Loops are possible; in the local smoke test Qwen3-1.7B
  looped until BFCL's step limit on one task.
- **LLM-jp's template writes today's date into the system prompt** unless given one. We fix
  `conversation_start_date` so all runs see the same prompt.
- **No system prompt from BFCL**: BFCL adds none in function-calling mode, so each model's
  template default applies (OLMo 3's is a function-calling prompt).
- **Tool schemas**: converted with BFCL's `convert_to_tool`; BFCL's non-standard `response`
  field is dropped. Same in both conditions.
- **`add_special_tokens: false`** on every request: the rendered prompt already has any BOS the
  template wants.
- **Argument names are checked** against the shown schema in both conditions; unknown ones reject the call. In the twin condition the backend would otherwise accept original parameter names silently.
- BFCL's `excluded_function` field is not used by BFCL's code; we don't use it either.
- Code and data licence of BFCL: Apache-2.0.

## Tools used

Code written with help from Claude Code (Anthropic).
