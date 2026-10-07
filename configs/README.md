Versioned training, evaluation, and preprocessing configurations go here.

`p7_baselines.json` records B0/A/P architecture, optimizer, masked loss, early
stopping and reproducibility settings. All three variants reuse one frozen P6
dataset version. The CPU default is portable; GPU runs enable CUDA AMP.

`contract_v0.example.json` records the portable manifest and per-sample
coordinate policy; it is an example, not a fixed training grid.

`p5_acoustics.json` records the locked 16 kHz / 80 ms emission-origin RIR
processor, peak normalization, nested bundle sizes and training augmentation.
Timing jitter is training-only (±0.2 ms); the low-pass range is 4–7.9 kHz.
P6 still defines splits and usable-K conditions.
