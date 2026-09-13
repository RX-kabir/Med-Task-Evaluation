# Medical Vision-Language Model Evaluation

A zero-shot pilot benchmark of two small open vision-language models on two medical imaging
tasks — one fine-grained image classification task, one closed-ended visual question
answering task.

The guiding question: **does a domain-specialized medical VLM actually beat a general-purpose
VLM of similar size?** The answer turns out to depend entirely on which task you ask about.

## Results

| | **Qwen2.5-VL-3B-Instruct**<br><sub>general-purpose</sub> | **MedGemma-4B-it**<br><sub>medical-specialized</sub> |
|---|---|---|
| **HAM10000** — 7-class skin lesion classification | **22.9%** <sub>(24/105)</sub> | 16.2% <sub>(17/105)</sub> |
| **VQA-RAD** — closed-ended yes/no radiology VQA | 68.0% <sub>(68/100)</sub> | **82.0%** <sub>(82/100)</sub> |

Random baseline on HAM10000 is 14.3% (1 of 7); on VQA-RAD it is 50%.

The ordering flips between the two tasks. The medical model loses on fine-grained
dermatological classification — landing barely above chance — and wins clearly on
closed-ended radiology VQA.

## Key findings

**HAM10000: both models collapse their output vocabulary.**
Neither model is meaningfully above the 7-class random baseline, and the reason is
visible in the prediction distributions rather than the headline number.

- **Qwen2.5-VL never predicts 3 of the 7 classes at all.** Across all 105 images it emits
  only `mel` (48×), `bkl` (26×), `nv` (25×) and `akiec` (6×). Every true `bcc`, `df` and
  `vasc` case is therefore wrong by construction.
- **MedGemma collapses onto melanoma even harder** — `mel` predicted 55 times out of 105,
  more than half of all responses, against 15 true melanoma cases.
- Both models score mel recall of 12/15. Much of the apparent accuracy is one class being
  over-predicted into correct answers by sheer volume, not genuine discrimination.

**A model-specific failure mode in MedGemma.**
7 of MedGemma's 105 responses could not be parsed. All 7 are the exact same string — `bcl`,
a class code that does not exist — spread across `df` (3), `vasc` (2), `bkl` (1) and `nv` (1).
This is a consistent, repeated non-answer rather than random format noise, and suggests a
specific confusion at the `bcc`/`bkl` class boundary. Worth re-testing at larger n.

**VQA-RAD: both models over-call findings as present.**
MedGemma leads on both sensitivity (86% vs 72%) and specificity (78% vs 64%), so its
advantage is not a threshold artifact. Both models lean toward answering "yes" when asked
whether a specific abnormality is present. Both miss the same pneumothorax case — a
clinically time-critical finding.

## Method

Both models are evaluated **zero-shot**: no fine-tuning, no few-shot examples, no
chain-of-thought prompting.

**HAM10000** (dermatoscopic skin lesions, 7 diagnostic classes)
- Stratified sample of 15 images per class = 105 images, `SEED = 42`
- The prompt lists all 7 class codes with their full names and asks for the code alone
- Scored by exact match against the reference label

**VQA-RAD** (radiology visual question answering)
- Train and test splits concatenated, filtered to the closed-ended subset
  (`answer ∈ {yes, no}`), then stratified to a balanced 50 yes / 50 no sample of 100 pairs,
  `SEED = 42`
- The prompt asks for the single word `Yes` or `No`
- Scored by exact match

**Comparability.** Within each task, the two models receive byte-identical prompts and
the identical sample, drawn by the same seeded sampling code. This was verified rather than
assumed: the two HAM10000 runs cover the same 105 image IDs in the same order, and the two
VQA-RAD runs cover the same 100 questions. Differences in results reflect the model, not the
inputs.

## Team

| Member | ID | Contribution |
|---|---|---|
| Monazir Mohammad Minhaz ([@Monajir](https://github.com/Monajir)) | 220041132 | VQA-RAD pipeline — Qwen2.5-VL |
| Rawad Hossain ([@rawadhossain](https://github.com/rawadhossain)) | 220041152 | VQA-RAD pipeline — MedGemma-4B |
| Mohammad Adnan Kabir ([@RX-kabir](https://github.com/RX-kabir)) | 220041160 | HAM10000 pipeline — both models (Qwen2.5-VL & MedGemma-4B) |

Each pipeline's commits are authored by the member who built it, so `git log` and the
Contributors graph reflect the split above.

## Repository layout

Each of the four runs is self-contained, with the same five artifacts:

```
HAM10000/
  MedGemma-4B/    notebook · sample CSV · results CSV · confusion matrix · findings report
  Qwen2.5/        notebook · sample CSV · results CSV · confusion matrix · findings report
VQA-RAD/
  MedGemma-4B/    notebook · sample CSV · results CSV · confusion matrix · findings report
  Qwen2.5/        notebook · sample CSV · results CSV · confusion matrix · findings report
```

| File | Contents |
|---|---|
| `*_pilot*.ipynb` | End-to-end notebook: data loading, sampling, inference, scoring |
| | All four notebooks retain their **stored outputs from the actual run**, so the scores, per-class report and confusion matrix can be read straight from the notebook without re-executing it |
| `*_sample*.csv` | The exact items sampled, so the run is auditable without re-sampling |
| `*_results*.csv` | Per-item ground truth, prediction, **and the raw model output** |
| `*_confusion_matrix*.csv` | Confusion matrix as scored |
| `*_Findings_Report.docx` | Written analysis: metrics, per-class breakdown, error analysis |

Every headline number in this README was recomputed directly from the `*_results*.csv`
files rather than copied from the reports.

## Reproducing

Notebooks are written for a Colab or Kaggle GPU runtime (T4 or better) and run top to bottom.

1. **HAM10000** — add the Kaggle dataset *Skin Cancer MNIST: HAM10000* (`kmader`), then set
   `DATA_DIR` in the notebook to its actual mount path.
2. **VQA-RAD** — loaded automatically from Hugging Face
   (`flaviagiammarino/vqa-rad`), no manual download needed.
3. **MedGemma runs only** — accept the model license at
   `huggingface.co/google/medgemma-4b-it` and supply an `HF_TOKEN` at the login cell.

No credentials are stored anywhere in this repository.

## Limitations

Stated plainly, because the sample sizes are small and the conclusions should be read
accordingly.

- **Small n.** At 105 and 100 items, the 95% confidence intervals are wide — roughly
  ±7 to ±9 percentage points. The HAM10000 intervals for the two models overlap, so that
  gap is directional, not established.
- **Per-class metrics on HAM10000 are not meaningful for every class.** Several classes have
  zero correct predictions, so their precision and recall cannot be distinguished from each
  other at this sample size. The qualitative finding — *which* classes collapse, and into
  what — is the durable result here, not the percentages.
- **Decoding is not fully controlled across models, and this was measured rather than
  assumed.** The MedGemma notebooks pass `do_sample=False` explicitly; the Qwen notebooks do
  not, so they inherit the model's default generation config, which is near-greedy but still
  nominally sampling. Re-running the Qwen HAM10000 notebook unchanged, on the identical 105-image
  sample, flipped 2 of 105 predictions — `ISIC_0030487` and `ISIC_0026656`, both `bkl` → `nv`,
  both incorrect either way. Overall accuracy was unchanged at 22.9%, and every qualitative
  finding above held, but the run is demonstrably **not bit-reproducible**. `do_sample=False`
  should be set on the Qwen notebooks before any full-scale comparison. The committed Qwen
  HAM10000 artifacts are all drawn from a single consistent run.
- **Label parsing is brittle by substring match.** `parse_label` returns the first class code
  appearing anywhere in the lowercased output, resolved in dictionary order. Every raw output
  in this pilot happened to be a bare single token, so it did not misfire here — but it would
  on longer or more verbose generations.
- **The question-type breakdowns in the VQA-RAD reports are keyword approximations**, not
  official VQA-RAD category metadata, which the public release does not retain. They are
  labeled as exploratory in the reports and several categories have very small n.

This is a pilot sized to validate the full analysis pipeline before a larger study, not to
produce precise accuracy estimates.
