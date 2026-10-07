Fast tests and small synthetic fixtures go here. Large dataset files stay outside Git.

`test_baselines.py` requires the optional `model` extra and tests P7 masking,
position controls, frozen sample integrity, checkpoint reload, deterministic
training and held-out isolation. CI runs it in a CPU PyTorch job and runs the
synthetic B0/A/P foundation gate. No dataset or checkpoint is shipped in Git.
`test_contract_v0.py` exercises the JSONL manifest, NPZ sample and prediction,
and metric RoomScene schema using only tiny synthetic fixtures.
