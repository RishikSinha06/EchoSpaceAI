"""Deterministic interventions: geometry, poses and valid pair identities stay fixed."""
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

CONDITIONS = ('correct', 'zero', 'swapped', 'noise20', 'gain_minus6', 'gain_plus6',
              'delay_plus3', 'delay_minus3', 'window40', 'window60', 'pose5cm')

class ControlledDataset(Dataset):
    def __init__(self, base, condition='correct', seed=0):
        if condition not in CONDITIONS:
            raise ValueError('unknown intervention')
        self.base, self.condition, self.seed = base, condition, seed
        self.entries = base.entries
        self.donors = {}
        if condition == 'swapped':
            # Match actual valid prefix length; require a different physical room group.
            counts = [int(base[i]['k']) for i in range(len(base))]
            for i, entry in enumerate(self.entries):
                group = entry['metadata']['room_group']
                candidates = [j for j,e in enumerate(self.entries)
                              if counts[j] == counts[i] and e['metadata']['room_group'] != group]
                if not candidates:
                    raise ValueError('no different-group donor with matching actual K')
                sid = entry['metadata']['sample_id']
                self.donors[i] = min(candidates,key=lambda j:hashlib.sha256(
                    f'{seed}|{sid}|{self.entries[j]["metadata"]["sample_id"]}'.encode()).digest())

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = {k:v.clone() for k,v in self.base[index].items()}
        sid = self.entries[index]['metadata']['sample_id']
        digest = hashlib.sha256(f'p11|{self.seed}|{sid}|{self.condition}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8],'little'))
        valid = item['rir_valid'].bool()
        wave = item['rir']
        if self.condition == 'zero':
            wave.zero_()  # valid pairs/poses retained: waveform ablation, not missing audio.
        elif self.condition == 'swapped':
            wave.copy_(self.base[self.donors[index]]['rir'])
        elif self.condition == 'noise20':
            noise = torch.from_numpy(rng.normal(size=tuple(wave.shape)).astype('float32'))
            rms = wave.square().mean(-1,keepdim=True).sqrt()
            noise_rms = noise.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)
            wave.add_(noise/noise_rms*rms/10)
        elif self.condition in ('gain_minus6','gain_plus6'):
            wave.mul_(10**((-6 if self.condition=='gain_minus6' else 6)/20))
        elif self.condition in ('delay_plus3','delay_minus3'):
            original = wave.clone(); wave.zero_()
            if self.condition=='delay_plus3': wave[:,3:] = original[:,:-3]
            else: wave[:,:-3] = original[:,3:]
        elif self.condition in ('window40','window60'):
            wave[:,640 if self.condition=='window40' else 960:] = 0
        elif self.condition == 'pose5cm':
            # Bounded independent pose error, accepted only within observed free cells.
            for key in ('src_pos','mic_pos'):
                for row in torch.where(valid)[0].tolist():
                    original = item[key][row].clone()
                    for _ in range(32):
                        candidate = original + torch.from_numpy(rng.uniform(-.05,.05,3).astype('float32'))
                        col,r = torch.floor(candidate[:2]/.2).long().tolist()
                        if 0<=col<64 and 0<=r<64 and item['observed_free'][r,col]:
                            item[key][row] = candidate
                            break  # otherwise keep original: do not invent valid observations.
        wave[~valid] = 0
        return item

    def donor_manifest(self):
        return [{'sample_id':self.entries[i]['metadata']['sample_id'],
                 'donor_sample_id':self.entries[j]['metadata']['sample_id'],
                 'donor_room_group':self.entries[j]['metadata']['room_group']}
                for i,j in sorted(self.donors.items())]
