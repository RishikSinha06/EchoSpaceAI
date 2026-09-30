# Local dataset selection

The user set a 50–60 GB overall storage cap and requested 10–20 GB for SoundSpaces/Replica. The original five routes are larger, so this folder holds representative subsets. Source archives and extracted data stay under ignored data/raw.

| Route | Selected local material | What it contains |
| --- | --- | --- |
| AcousticRooms | Existing OBJ mesh checkout; main single-channel RIR, per-pair metadata and receiver depth archives | Simulated room meshes, acoustic responses, source/receiver positions, receiver observations. The large raw-dense simulation release is outside this selection. |
| DIFFRIR | Classroom base ZIP and official room-geometry repository | Real mono/binaural RIR arrays, music recordings, measured mic locations, room surfaces and speaker location. This is one of 14 configurations across four rooms. |
| RAF | Furnished-room split RIR ZIP and its OBJ/MTL/JPG mesh | Real measured RIRs, transmitter/receiver poses, and a reconstructed textured room mesh. Empty-room recordings are outside this selection. |
| FLAIR | data_FLAIR.mat | One measured room, RIRs and calibrated point-cloud/position data. |
| SoundSpaces + Replica | Replica scene metadata and five selected binaural-RIR archives: office_0, room_0, room_1, hotel_0 and apartment_0 | Simulated two-channel WAV RIRs indexed by source, receiver and heading; navigation-graph positions. The apartment_0 RIRs pair with the extracted apartment_0 Replica mesh and Habitat navmesh. |

The SoundSpaces selected archives total 17,138,611,129 bytes compressed. They are a scene-based subset of the much larger official release. The apartment_0 mesh (200,778,055 bytes; 4,564,613 vertices) and navmesh were extracted from the separate Replica release. A small office_0 archive is also extracted for inspection. The viewer and research model should not pool these five sources until units, coordinate frames, licenses, room identity and RIR-to-geometry pairing are validated.

## Official distribution pages

- AcousticRooms: https://github.com/facebookresearch/AcousticRooms
- DIFFRIR: https://zenodo.org/records/11195833 and https://github.com/maswang32/hearinganythinganywhere
- RAF: https://github.com/facebookresearch/real-acoustic-fields
- FLAIR: https://zenodo.org/records/17037517
- SoundSpaces: https://github.com/facebookresearch/sound-spaces/blob/main/soundspaces/README.md
- Replica: https://github.com/facebookresearch/Replica-Dataset

## Coordinate frames and archive verification

RAF explicitly uses X forward, Y up, Z left, meters. Do not apply the AcousticRooms viewer orientation to RAF or Replica without checking each source frame. FLAIR data_FLAIR.mat has RIR shape (17,873 samples, 135 receivers, 2 speakers) and 2,924,201 boundary points.

## Archive verification

- AcousticRooms Git LFS pointers give SHA-256 values for the main RIR, metadata and depth archives.
- Zenodo JSON manifests in data/raw/_manifests contain FLAIR and DIFFRIR checksums and sizes.
- RAF's S3 listing in data/raw/_manifests contains exact multipart object sizes.
- SoundSpaces scene archive sizes were checked with official HTTP headers. The office_0 tar was opened and contains 2,704 WAV RIRs; sample files are 44.1 kHz, stereo, 32-bit float WAVs.

Do not treat an incomplete .download file or a 135-byte Git LFS pointer as an acquired archive.
