# Data licences

The EchoSpace AI **source code** is MIT licensed (see `LICENSE`). Data used by
the project carries its own, separate obligations. They apply to raw archives
and to everything derived from them: index caches, manifests, occupancy and
wall labels, sample caches, figures and the redistributed sample in
`data/samples/`.

## AcousticRooms

- Source: <https://github.com/facebookresearch/AcousticRooms> (Meta Platforms, Inc.)
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
  <https://creativecommons.org/licenses/by/4.0/>
- Our use: meshes, per-pair source/receiver metadata and single-channel RIRs
  are read in place from the original archives. Derived artifacts (manifests,
  audit figures, rasterized labels, cached features) are adaptations and must
  keep this attribution and state that changes were made.
- Redistribution: a small unmodified subset (1–2 rooms, a few source/receiver
  pairs) lives in `data/samples/` with its own `ATTRIBUTION.md`. No other raw
  AcousticRooms file is committed to this repository.

Required citation:

```bibtex
@inproceedings{liu2025haae,
  title     = {Hearing Anywhere in Any Environment},
  author    = {Liu, Xiulong and Kumar, Anurag and Calamia, Paul and
               Gar{\'i}, Sebasti{\`a} V. Amengual and Murdock, Calvin and
               Ananthabhotla, Ishwarya and Robinson, Philip and
               Shlizerman, Eli and Ithapu, Vamsi Krishna and Gao, Ruohan},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

## Other datasets

DIFFRIR, RAF, FLAIR and SoundSpaces/Replica are not used by the H1 pipeline.
Each has its own licence, which must be recorded here before any derived
artifact from it is committed or published.
