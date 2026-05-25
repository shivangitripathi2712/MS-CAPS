# MS-CAPS: Minimal Sufficient Image Captions for Safety-Critical Traffic Scene Understanding

In safety-critical traffic scene understanding, a caption is useful only if it
preserves the details that matter — vehicles, signs, plates, and their spatial
relations — while discarding verbose, redundant description. MS-CAPS produces
**minimal sufficient captions**: the shortest caption that still preserves the
visual semantics of the scene, under a reconstruction-guided rate–distortion
criterion.

The key idea is to measure caption sufficiency by *reconstruction*. A
text-to-image model regenerates an image from a candidate caption; a knowledge
graph (KG) extracted from that reconstruction is compared against the KG of the
original image. The discrepancy is the semantic **distortion**. A caption is
compressed only as far as a distortion gate allows, so safety-relevant content
is retained while redundancy is removed.

## Method

Given an image *I* and its dense caption *c*:

1. **Reference semantics.** Extract a scene-graph `KG_I` (objects, attributes,
   relations) and visible text (plates, signs, brands) from the original image
   *I*. This defines the semantics that must survive compression.

2. **Anchor derivation.** From *c* and `KG_I`, derive:
   - **Strict anchors** — verbatim content that must be kept (visible text such
     as license plates and signage), and
   - **Soft anchors** — salient objects and relations that should be preserved.

3. **Candidate generation.** A rewriter proposes shorter captions across a
   ladder of token budgets, conditioned on the strict and soft anchors. If no
   candidate at the primary budgets passes, the budgets are relaxed
   (escalation).

4. **Reconstruction-guided distortion.** Each candidate `c'` is reconstructed
   to an image `G(c')` with a text-only text-to-image model. Distortion is
   measured at three levels:
   - **Q_img** — image similarity between *I* and `G(c')` (CLIP-I, DINOv2,
     LPIPS),
   - **Q_cap** — caption fidelity (BERTScore, CLIPScore),
   - **Q_KG** — KG agreement: the KG of `G(c')` vs `KG_I` (object, attribute,
     and relation recall),
   combined into a single distortion `D`.

5. **Distortion gate.** A candidate is accepted only if every quality condition
   holds (Q_img, Q_cap, Q_KG above their thresholds) and `D` stays within the
   baseline-derived budget.

6. **Selection.** Among accepted candidates, MS-CAPS selects the **most
   compressed** caption that remains faithful — the minimal sufficient caption.

## Ablations

The repository includes the two controlled studies reported in the paper:

- **Compressor ablation** (`run_compressor_comparison.py`): swap the caption
  compressor (Qwen / GPT-4o / Gemini) while holding the reconstruction and KG
  evaluator fixed, isolating the effect of the rewriter.
- **Evaluator ablation** (`evaluate_gpt4o.py`): swap the KG evaluator (GPT-4o as
  scene-graph judge) while holding the pipeline fixed, testing whether results
  depend on the choice of KG extractor.

## Repository layout

```
run_pipeline.py                 main MS-CAPS pipeline over a dataset
run_compressor_comparison.py    compressor ablation (Qwen / GPT-4o / Gemini)
evaluate_gpt4o.py               evaluator ablation (GPT-4o as KG judge)
rewriters_api.py                GPT-4o / Gemini rewriter backends
kg_anchors.py                   anchor-derivation utilities
config.yaml                     run settings (paths, thresholds, budgets)
requirements.txt                Python dependencies
setup.sh                        environment setup helper
src/
  dataset.py                    image + dense-caption loader
  scene_graph.py                VLM KG extraction, comparison, anchors
  t2i_reconstruct.py            text-to-image reconstruction G
  distortion.py                 Q_img / Q_cap / Q_KG, distortion D, gate
  compressor.py                 core compressor (Stage 1 + Stage 2, selection)
  parse_tree.py                 caption parsing
  token_importance.py           token-level importance
  evaluate.py                   metrics, CSVs, tables
```

## Setup

```bash
conda create -n mscaps python=3.10 -y
conda activate mscaps
pip install -r requirements.txt
# optional spaCy model (a regex fallback is used if absent):
python -m spacy download en_core_web_sm
```

A CUDA GPU is required: the reconstruction model and the VLM KG extractor run
on GPU.

## Data

The image / dense-caption dataset is **not** included in this repository. Place
data under `inputs/` in the layout expected by `src/dataset.py`:

```
inputs/
  <dataset>/
    images/      # image files
    captions/    # one .txt caption per image, matched by filename stem
```

Set the corresponding paths in `config.yaml`.

## Running

Main pipeline:

```bash
python run_pipeline.py --config config.yaml --max-samples 50
```

Compressor ablation (needs API keys for the hosted rewriters):

```bash
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export GEMINI_API_KEY=<YOUR_GEMINI_KEY>
python run_compressor_comparison.py --config config.yaml --max-samples 50 \
    --models qwen gpt4o gemini
```

Evaluator ablation (run **after** the main pipeline, on its result JSONs):

```bash
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
python evaluate_gpt4o.py --config config.yaml
```

## Notes

- API keys are read from environment variables; none are stored in the repo.
- Cache and output directories are created locally at run time.
- Per-sample results, reconstructions, and reports are written under `outputs/`.