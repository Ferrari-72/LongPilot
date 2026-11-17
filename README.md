# LongPilot memory reliability pilot

Compare lexical retrieval of raw dialogue turns, Mem0 retrieval, and Mem0 retrieval with a timestamp and provenance heuristic. The three routes share an answer model, instruction prompt, and evidence budget. A separate offline stress test exercises stale facts and conflicting records.

## Environment

The live pilot ran on Windows with Python 3.13.2, Mem0 OSS 2.2.1, local Qdrant, and SiliconFlow. The offline experiment uses the Python standard library. For the live experiment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data and selected cases

Download the September 2025 cleaned LongMemEval-S dataset from the [official LongMemEval repository](https://github.com/xiaowu0162/LongMemEval) and save it in the repository root as `longmemeval_s_cleaned.json`. Verify SHA-256:

```powershell
Get-FileHash .\longmemeval_s_cleaned.json -Algorithm SHA256
```

Expected hash: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.

`selected-ids.tsv` lists the **22 completed cases** in run order with the original question types. The sampler uses seed 42, rotates across question types and an abstention stratum, and sorts each case's sessions chronologically. The original plan selected 24 cases. A network interruption left the final two cases incomplete. The instructions below use 22 to reproduce the sample reported in the pilot.

## Offline diagnostics

```powershell
.\.venv\Scripts\python.exe run.py --dataset longmemeval_s_cleaned.json --limit 22 --output-dir results/offline
```

The script writes `synthetic.jsonl` and `benchmark_retrieval.jsonl`. To run just the synthetic test, omit `--dataset`. The synthetic data has six independent fields with old and new user facts and an external instruction. Each field has current, historical, and unanswerable queries. The two routes use one evidence record and the same deterministic reader. Plain retrieval scored 12/18 and the time-and-source variant scored 18/18. These outputs are separate from the live benchmark. The offline benchmark retrieves evidence under a 2,000-word budget and does not generate answers.

## Live model comparison

Use PowerShell 7 and enter your SiliconFlow API key when prompted. Run from the repository root.

```powershell
$env:SILICONFLOW_API_KEY = Read-Host -MaskInput 'SiliconFlow API key'
.\.venv\Scripts\python.exe -u run_model.py --provider siliconflow --dataset longmemeval_s_cleaned.json --limit 22 --output results/live-22.jsonl
```

The default SiliconFlow endpoint is `https://api.siliconflow.cn/v1`. The chat model for answering and memory extraction is `Qwen/Qwen2.5-7B-Instruct`, and the embedding model is `BAAI/bge-m3`. Override with `SILICONFLOW_BASE_URL`, `--model`, `--memory-model`, or `--embedding-model`. With `--provider openai`, set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`. The live runner sends public benchmark dialogue to the configured provider.

Start with a fresh output path. For a smoke test, use `--limit 1` and a separate filename. To continue an interrupted 22-case run, repeat the live command with the same settings and add `--resume`. Resume checks that the output is a complete, ordered prefix of the same experiment. A case is appended after all three answers are generated.

- **Plain:** ranks raw dialogue turns by inverse-frequency-weighted query term overlap.
- **Mem0:** writes sessions chronologically with one user ID per case and retrieves up to 50 derived memories. The pilot ran with BM25 disabled because spaCy and fastembed were absent.
- **Mem0 + filter:** reuses the same Mem0 search results. For explicitly current-state questions, it drops candidates without source or date metadata and sorts the first five eligible items by session date. Session date approximates event time.

The default evidence budget is 2,000 tokens per route. For the SiliconFlow reader, the runner uses the same `o200k_base` proxy tokenizer for all routes. Inspect `evidence[].line` for the exact text sent to the reader. Reader usage counts and Mem0 SDK write/search counts are logged separately. SDK calls do not measure underlying API calls or total token cost.

## Pilot results

The completed sample contains 18 answerable and 4 unanswerable cases. Manual review found 4/18, 8/18, and 7/18 correct answers on answerable cases for Plain, Mem0, and Mem0 + filter, respectively. Correct abstentions were 3/4, 4/4, and 4/4. Support-session hits were 11/18, 18/18, and 18/18. That metric checks whether any annotated support session appeared in the evidence and does not establish whether the required fact was present or understood. The filter changed the actual model input for one case in this sample. No official model-judge score is reported. For official scoring, export each route as JSON Lines with `question_id` and `hypothesis` and follow the [LongMemEval evaluation instructions](https://github.com/xiaowu0162/LongMemEval#evaluation).
