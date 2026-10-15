"""Verified memory-mapped sidecar for p8 freezes; no sample values change."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .audio_frozen import ARRAY_KEYS, FORMAT, sample_sha, validate_audio_item
from .frozen import json_sha


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pack_freeze(manifest_path, destination, progress=None):
    source = Path(manifest_path)
    manifest = json.loads(source.read_text(encoding='utf-8'))
    if manifest.get('format') != FORMAT or manifest['checksum'] != json_sha({k:v for k,v in manifest.items() if k!='checksum'}):
        raise ValueError('invalid source frozen manifest')
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise ValueError('packed destination must be empty')
    root.mkdir(parents=True, exist_ok=True)
    entries = manifest['samples']
    if not entries:
        raise ValueError('empty source freeze')
    arrays, specs = {}, {}
    for i, entry in enumerate(entries):
        path = (source.parent/entry['path']).resolve()
        if not path.is_relative_to(source.parent.resolve()):
            raise ValueError('sample path escapes source')
        with np.load(path, allow_pickle=False) as data:
            item = {key:data[key].copy() for key in ARRAY_KEYS}
        if sample_sha(entry['metadata'], item) != entry['sha256']:
            raise ValueError('source sample checksum mismatch')
        validate_audio_item(item, entry['metadata'])
        for key, value in item.items():
            if i == 0:
                shape = (len(entries), *value.shape)
                arrays[key] = np.lib.format.open_memmap(root/f'{key}.npy', mode='w+', dtype=value.dtype, shape=shape)
                specs[key] = {'path':f'{key}.npy', 'shape':list(shape), 'dtype':str(value.dtype)}
            if list(value.shape) != specs[key]['shape'][1:] or str(value.dtype) != specs[key]['dtype']:
                raise ValueError('inconsistent packed sample shape/dtype')
            arrays[key][i] = value
        if progress and (i+1)%512==0:
            progress({'stage':'packing', 'samples':i+1, 'total':len(entries)})
    for array in arrays.values():
        array.flush()
    arrays.clear()
    for spec in specs.values():
        spec['sha256'] = file_sha(root/spec['path'])
    record = {'format':'p8.packed.1', 'frozen_checksum':manifest['checksum'], 'samples':len(entries), 'arrays':specs}
    record['checksum'] = json_sha(record)
    (root/'manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    return root/'manifest.json'


def open_packed(directory, frozen_checksum, count):
    root = Path(directory)
    meta = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if meta.get('format')!='p8.packed.1' or meta['checksum']!=json_sha({k:v for k,v in meta.items() if k!='checksum'}):
        raise ValueError('packed manifest integrity mismatch')
    if meta['frozen_checksum']!=frozen_checksum or meta['samples']!=count or set(meta['arrays'])!=set(ARRAY_KEYS):
        raise ValueError('packed source identity mismatch')
    arrays = {}
    for key,spec in meta['arrays'].items():
        path=(root/spec['path']).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha(path)!=spec['sha256']:
            raise ValueError('packed array path/checksum mismatch')
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        if list(array.shape)!=spec['shape'] or str(array.dtype)!=spec['dtype'] or len(array)!=count:
            raise ValueError('packed array shape/dtype mismatch')
        arrays[key]=array
    return arrays
