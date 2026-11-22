# MS-CAPS: Minimal Sufficient Image Captions for Safety-Critical Traffic Scene Understanding

Code for the paper **"MS-CAPS: Minimal Sufficient Image Captions for Safety-Critical Traffic Scene Understanding"** (Findings of EACL 2027).

Long image captions often repeat themselves or include details that do not matter. MS-CAPS shortens a caption while keeping what matters in traffic scenes: vehicles, signs, license plates, and where things are relative to each other.

The main idea is to **test a short caption by redrawing the image from it**. A text-to-image model (FLUX.1-dev) generates an image from the short caption. We then compare that image with the original. If the important content is still there, the short caption is good enough. MS-CAPS keeps the shortest caption that passes this check.

## How it works

For each image *I* and its long caption *c*:

1. **Build a scene graph of the original image.** Qwen2.5-VL lists the objects, attributes, and relations in *I*, plus any readable text (plates, signs). We call this graph `KG_I`.
2. **Set a baseline.** FLUX redraws the image from the full caption *c*. We measure how far this redrawn image is from *I* (the baseline distortion `D_0`). The short caption may be at most a little worse than this (`delta = min((1 + epsilon) * D_0, absolute_threshold)`).
3. **Choose what must be kept.**
   - *Strict anchors*: exact text that must appear word for word, such as a license plate.
   - *Soft anchors*: important facts that must still be said, in any wording.
4. **Write shorter versions.** A rewriter (Qwen2.5-VL by default) writes several shorter captions at different target lengths (35% to 70% of the original).
5. **Score each short caption.** FLUX redraws an image from it, and we compute:
   - `Q_img`: how similar the redrawn image is to *I* (CLIP, DINOv2, LPIPS),
   - `Q_cap`: how close the short caption is to the original (BERTScore, CLIPScore),
   - `Q_KG`: how many objects, attributes, and relations of `KG_I` appear in the scene graph of the redrawn image,
   - `D`: one combined distortion score.
6. **Keep the shortest caption that passes all four checks:** `Q_img >= tau_img`, `Q_cap >= tau_cap`, `Q_KG >= tau_kg`, and `D <= delta`. If none pass, we try gentler lengths (75% to 95%), and then fall back to the best shorter caption.

All settings used in the paper are in [`config.yaml`](config.yaml).

## Repository layout

```
run_pipeline.py                main MS-CAPS run over a dataset
run_compressor_comparison.py   ablation: swap the rewriter (Qwen / GPT-4o / Gemini)
evaluate_gpt4o.py              ablation: re-score results with GPT-4o as the scene-graph extractor
llm_rewriters.py               GPT-4o and Gemini rewriters (same prompt as Qwen)
config.yaml                    all settings and thresholds
requirements.txt               Python packages
setup.sh                       one-step install
src/
  compressor.py                the MS-CAPS loop: anchors, candidates, gate, selection
  scene_graph.py               scene-graph extraction (Qwen2.5-VL) and graph comparison
  t2i_reconstruct.py           FLUX.1-dev image reconstruction
  distortion.py                Q_img, Q_cap, Q_KG, distortion D, and the gate
  parse_tree.py                strict-anchor extraction from the caption
  dataset.py                   loads (image, caption) pairs from folders
  evaluate.py                  metrics tables and CSVs
```

## Installation

```bash
conda create -n mscaps python=3.10 -y
conda activate mscaps
bash setup.sh
```

FLUX.1-dev is a gated model. Accept its license on its [Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-dev) and log in before the first run:

```bash
huggingface-cli login
```

**Hardware.** The default config uses two GPUs: `cuda:0` for Qwen2.5-VL-7B and the metrics, and `cuda:1` for FLUX.1-dev. To change this, edit the `device` fields in `config.yaml`.

## Data

Images and captions are not included in this repository. We used three traffic datasets (50 images each):

- Indian Vehicle Dataset (DataCluster Labs)
- [CODA-LM](https://coda-dataset.github.io/coda-lm/)
- NuInteract

Put each dataset in two folders: one with images, one with captions. Each caption is a `.txt` file with the same name as its image:

```
inputs/
  vehicle_image_dataset/
    indian_vehicle_images/indian_vehicle_images/   # 0001.jpg, 0002.jpg, ...
    generated_captions/                            # 0001.txt, 0002.txt, ...
```

Then point `config.yaml` to the folders:

```yaml
dataset:
  local:
    data_dir: "./inputs"
    images_subdir: "vehicle_image_dataset/indian_vehicle_images/indian_vehicle_images"
    captions_subdir: "vehicle_image_dataset/generated_captions"
```

To run on another dataset, change these three paths (and `paths.output_dir`, so results stay separate). Captions with fewer than 30 words are skipped (`min_caption_tokens`).

## Running

**Main pipeline** (50 images):

```bash
python run_pipeline.py --config config.yaml --max-samples 50
```

Finished images are skipped if you re-run the command. Use `--no-resume` to recompute them.

**Rewriter ablation** (Qwen2.5-VL vs. GPT-4o vs. Gemini 2.5 Pro). Only the rewriter changes; everything else stays fixed.

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
python run_compressor_comparison.py --config config.yaml --max-samples 50 \
    --models qwen gpt4o gemini
```

**Evaluator ablation** (GPT-4o as the scene-graph extractor). Run this after the main pipeline; it re-scores its outputs.

```bash
export OPENAI_API_KEY=...
python evaluate_gpt4o.py --output-dir outputs
```

API keys are read only from environment variables. None are stored in this repository.

## Outputs

```
outputs/
  results/<image_id>.json        final caption, scores, anchors, every candidate step
  results/metrics.csv            per-image metrics
  results/metrics_summary.csv    averages
  results/summary.json           run summary (status counts, compression, distortion)
  kg_rd_report.xlsx              Excel report with images side by side
  reconstructions/<image_id>/    FLUX images for the baseline and the candidates
  kg_debug/  candidates/  debug_samples/   intermediate files (if save_intermediate: true)
  compressor_<model>/            per-rewriter outputs from the rewriter ablation
  compressor_comparison/         per_sample.csv and summary.csv for the rewriter ablation
  results_gpt4o/                 per-image GPT-4o scores from the evaluator ablation
```

Each image ends with one of these statuses:

- `compressed`: a shorter caption passed all checks.
- `compressed_via_fallback`: no caption passed, so the best shorter one was kept.
- `skipped_low_baseline_fidelity`: even the full caption could not be redrawn well, so the image was skipped.
- `kept_original_rewriter_failed`: the rewriter returned no usable caption.

## Citation

If you use this code, please cite our EACL 2027 Findings paper. The BibTeX entry will be added here once the proceedings are online.

## License

This code is released under the [MIT License](LICENSE). The models (Qwen2.5-VL, FLUX.1-dev) and datasets have their own licenses; please check them before use.
