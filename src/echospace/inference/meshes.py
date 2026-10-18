"""Serve audited source OBJ bytes through P2, without extracting ZIP members."""
import hashlib
import json
from pathlib import Path, PurePosixPath

from echospace.io.adapters.acousticrooms import ArchiveReader, MemberRef


class MeshStore:
    def __init__(self, root, manifest):
        self.root = Path(root) if root else None
        self.entries = {}
        self.issue = None
        if self.root is None or not self.root.is_dir():
            self.issue = 'Configure ECHOSPACE_ACOUSTICROOMS_ROOT with the local dataset folder.'
            return
        if not Path(manifest).is_file():
            self.issue = 'Audited AcousticRooms mesh inventory is unavailable.'
            return
        for line in Path(manifest).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get('mesh_path'):
                continue
            chain = tuple(row['mesh_path'].split(' :: '))
            # The same audited member may already be extracted in a repo checkout.
            loose = f"room_mesh_obj_format/{row['category']}/{row['room_id']}.obj"
            if not (self.root/chain[0]).is_file() and (self.root/loose).is_file():
                chain = (loose,)
            if not (self.root/chain[0]).is_file():
                continue
            if any(PurePosixPath(p.replace('\\','/')).is_absolute() or '..' in PurePosixPath(p.replace('\\','/')).parts for p in chain):
                raise ValueError('audited mesh path is unsafe')
            if not (self.root/chain[0]).resolve().is_relative_to(self.root.resolve()):
                raise ValueError('mesh container escapes dataset root')
            identity = row['category']+'/'+row['room_id']
            self.entries[identity] = {'id':identity,'room_id':row['room_id'],'name':row['room_id'],
                                      'category':row['category'],'bytes':row.get('mesh_bytes'),
                                      'storage':'archive' if len(chain)>1 else 'directory','ref':MemberRef(chain)}
        if not self.entries:
            self.issue = 'No audited meshes found in the configured dataset folder.'

    def catalogue(self):
        return {'available':bool(self.entries),'issue':self.issue,'rirAvailable':False,
                'source_frame':'AcousticRooms: metres, Z up; source_to_scene=(x,z,-y)',
                'rooms':[{k:v for k,v in e.items() if k!='ref'} for _,e in sorted(self.entries.items())]}

    def read(self, identity):
        entry = self.entries.get(identity)
        if entry is None:
            raise LookupError('Unknown audited mesh id')
        with ArchiveReader(self.root) as reader:
            data = reader.read(entry['ref'])
        if len(data)>64*1024*1024:
            raise ValueError('Mesh exceeds local viewer size limit')
        return data, hashlib.sha256(data).hexdigest()
