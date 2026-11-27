# The Undo Problem: Do LLM Agents Know Which Actions Cannot Be Undone?

**Hanish Reddy** (SE23UARI111) · **Aishika Reddy** (SE23UARI075)
Artificial Deep Neural Networks (ADNN), 2026 · Mahindra University

We test whether an LLM agent (Qwen2.5-7B-Instruct) represents, states and acts on whether an action is reversible,
using a tool sandbox where reversibility is computed by searching for undo sequences (2,400 labelled pairs, 75% minimal pairs).

## Key results

| Question | Result |
|---|---|
| RQ1: decodable from hidden states? | Yes: 0.93 balanced accuracy at layer 20; 0.93 on minimal pairs vs 0.58 for bag-of-words |
| RQ2: does the model say it correctly? | No: 0.40 accuracy (3-class) vs 0.86 for the probe; calls 93% of reversible actions risky |
| RQ3: does the agent act more carefully? | No: 0% caution in both versions; took the shortcut in 100% of runs (30 task pairs) |
| Knowing–acting gap K − A | **0.83** (95% CI 0.70–0.97) |
| RQ4: does steering increase caution? | No: 0% at +0.3, same as a random direction |
| RQ5: does uncertainty signal irreversibility? | No: AUROC 0.46 (0.50 within verb) |

## Files

| File | Contents |
|---|---|
| `of_ADNN_Irreversibility_vfinal.ipynb` | the full pipeline (sandbox, dataset, probes, agent runs, steering, analysis) |
| `The Undo Problem ... .pdf/.docx` | the project report |
| `reversibility_dataset.jsonl` | the 2,400 labelled (state, action) pairs |
| `ADNN_datasets_and_results.zip` | `data/` (dataset + data card), `results/` (all tables, agent logs, figures), `requirements.txt` |

## Reproduce

Open the notebook in Google Colab with a GPU (T4 is enough) and run the cells top to bottom.
The hidden-state files (~1 GB) are not included because of GitHub's size limit; the notebook regenerates them (~15 min on a T4).

Compute-limited settings used: RQ2 on 600 pairs (full test split of 468 + 132 random), RQ3 on 30 task pairs with a 6-step limit,
RQ4 on 12 pairs at one layer; ScienceWorld and the second model family were not run.
