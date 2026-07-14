# VTRACE max-quality runbook

The production profile is [`config/nht_max.yaml`](config/nht_max.yaml): NVIDIA 3DGRUT, distorted-camera 3DGUT, MCMC densification, and 48-feature Neural Harmonic Textures (NHT), full resolution, 30k iterations, and a 4M primitive cap.

This is the highest-quality official implementation selected for this repository. It is not a promise of a particular VTRACE PSNR: the public scene smoke/full runs must measure that. NVIDIA's current official NHT validation uses 1M primitives; this profile spends a larger 4M budget to favor quality and therefore requires substantially more memory and storage.

## Server

- Recommended: one A100/H100 80GB. Minimum accepted by the preflight: L40/L40S/RTX A6000/RTX 6000 Ada 48GB.
- At least 64GB system RAM and 600GB free persistent disk. NHT optimizer checkpoints at 4M primitives are very large.
- Keep persistent storage mounted. A stopped ephemeral instance cannot be recovered by application code.
- CUDA 12.8 is the safest choice for current 3DGRUT, especially on Blackwell GPUs.

## One-time setup

```bash
chmod +x scripts/setup_all.sh scripts/launch_nht_max.sh
./scripts/setup_all.sh
```

`setup_all.sh` installs both the repository environment and an isolated 3DGRUT/NHT environment, downloads data when absent, and checks out commit `a37ef721012dea0f29c0fcfff2d525023b4e854a`; the runner refuses any other revision.

## Mandatory paid-server smoke test

Use one public scene. This compiles the CUDA extensions and verifies train, checkpoint, exact test-pose normalization, render, image names, dimensions, and ZIP creation with only ten iterations:

```bash
./scripts/launch_nht_max.sh \
  --smoke-test \
  --data-dir VAI_NVS_DATA/phase1/public_set \
  --scene HCM0181
tail -f output_nht_smoke/launcher.log
```

Do not start the expensive run until `output_nht_smoke/DONE.json` exists.

## Full public validation

```bash
VTRACE_NHT_RUN_DIR="$PWD/output_nht_max_public" \
./scripts/launch_nht_max.sh \
  --data-dir VAI_NVS_DATA/phase1/public_set \
  --output-dir output_nht_max_public
```

After completion, evaluate public ground truth:

```bash
uv run python scripts/evaluate_public.py \
  --data-dir VAI_NVS_DATA/phase1/public_set \
  --prediction-dir output_nht_max_public/submission \
  --output-dir output_nht_max_public/evaluation
```

## Full private run

The default config already targets the private set:

```bash
./scripts/launch_nht_max.sh
```

Safe monitoring commands:

```bash
tail -f output_nht_max_private/launcher.log
tail -f output_nht_max_private/logs/HCM0249.train.log
find output_nht_max_private/scenes -name status.json -print
```

If SSH disconnects, `setsid`/`nohup` keeps the job alive. If the provider reboots or preempts the instance, run the same launch command again: completed scenes are skipped and interrupted scenes resume from the newest official checkpoint. The runner never treats a failed subprocess as success, never renders a dummy model, and only creates `DONE.json` after strict submission validation.

No program can guarantee survival if the provider deletes the instance or its disk. Persistent storage plus the checkpoint/resume mechanism is the protection for that case.
