# Familiar or capable?

Does renaming tools to unfamiliar but equally clear names change multi-turn tool-calling
results? A paired pilot on BFCL `multi_turn_base`.

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
on (b, c), direction fixed by H1 (for reference: 5–0 gives p = 0.031, 6–1 gives 0.063,
7–1 gives 0.035). Intrusions = twin-condition calls to an original *function* name not in the tool
list (original *argument* names are counted separately and are not part of H3). "Models" =
those passing every validity gate below, including Floor. Rows are checked top to
bottom; the first match is the outcome. Implemented in `src/decide.py`.

| # | Outcome | Condition | Wording allowed |
|---|---|---|---|
| 1 | Strong signal | p < 0.05 in ≥ 1 model **and** net ≥ 3 in a second model | "a consistent drop" |
| 2 | Suggestive | net ≥ 3 in ≥ 2 models | "suggestive, not significant at n = 20" |
| 3 | Reversed | net ≤ −3 in ≥ 2 models | "twin names did better"; re-check the clarity test before drawing anything from it |
| 4 | Behaviour without score change | net ≤ 2 in every model, and H3 holds in ≥ 1 model | "familiarity changes behaviour, not score" |
| 5 | Null | \|net\| ≤ 1 in every model, and H3 fails in every model | "no effect found on this benchmark" |
| 6 | Inconclusive | anything else | "inconclusive" |

If fewer than 2 models pass the gates, rows 1–3 cannot match; report per-model counts.

H3 holds for a model if it makes ≥ 3 intrusions across ≥ 2 tasks, more than its calls to
non-existent tools in the original condition. H2 is assessed only with ≥ 5 original-pass /
twin-fail tasks pooled across models, and holds if ≥ 60% of them fail first on tool choice
(wrong tool or intrusion) rather than arguments.

**Validity gates.** Results count only if all pass.

| Gate | Pass condition | If it fails |
|---|---|---|
| Replay test | Ground-truth calls replayed through twin names pass the checker on 20/20 tasks | Fix the mapping layer; do not run models |
| Intrusion blocking | A deliberate original function name and a deliberate original argument name in the twin condition are each rejected and logged, and neither reaches the backend | Fix before running |
| Clarity check | A judge model, shown only the name and parameter names, describes the tool correctly for ≥ 90% of twin names, and the twin rate is no more than 5 points below the original rate | Rename and re-check |
| Parser parity | Parse-failure rate in the twin condition is within 5 points of the original condition, or the difference is explained by logged intrusions | Inspect raw outputs; fix the parser if it is format-related |
| Parser health | In the original condition, under 10% of a model's steps fail to parse | Fix the template or parser, or drop that model and report why |
| Truncation | Under 2% of steps stop on `length` | Raise `max_tokens`, rerun that model |
| Floor | Each model passes at least 5/20 tasks in the original condition | Report it, but leave it out of the decision rule |
| Run-to-run stability | Repeating 5 tasks × 2 conditions on one model changes at most 1 of 10 outcomes | Run 3 repeats for all and report the spread |
| Traceability | Every scored call traces back to a logged prompt and raw output | Rerun the untraced parts |

**Tasks.** 20 of 200 `multi_turn_base` tasks, drawn by `scripts/select_tasks.py` with seed
20261114, stratified by main backend: 8 file system, 4 vehicle control, 4 trading, 4 travel.
IDs in `tasks/selected_ids.json`; strata and reserve order in `tasks/selection.json`. No
outcome-based exclusions. A reserve replaces a task only for a pipeline reason found before
any model run, recorded in `tasks/selection.json`. All 20 pass the ground-truth replay in the
original condition.

Still to freeze before the first real run: name mappings and clarity check, model
revisions, vLLM version and sampling settings.

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
