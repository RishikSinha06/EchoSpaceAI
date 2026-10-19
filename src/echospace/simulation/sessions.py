"""Bounded local session state, discrete virtual-clock events and portable bundles."""
from collections import OrderedDict
from dataclasses import dataclass,field
import io,json,threading,uuid,zipfile
import numpy as np
from . import ENGINE_VERSION
from .config import from_dict
from .engine import virtual_capture,replay_capture

@dataclass
class Session:
    identifier: str
    config: object
    anchor: tuple
    heading: float
    state: str='configured'
    revision: int=0
    clock_s: float=0.
    ordinal: int=0
    last_capture: str | None=None
    events: list=field(default_factory=list)
    history: list=field(default_factory=list)
    event_sequence: int=0
    def event(self,kind,time_s=None,**data):
        self.events.append({'sequence':self.event_sequence,'type':kind,'time_s':round(self.clock_s if time_s is None else time_s,9),**data})
        self.event_sequence+=1
        self.events=self.events[-256:]
    def public(self):
        return {'format':'echospace.simulation_session.v1','session_id':self.identifier,'state':self.state,
                'revision':self.revision,'clock_s':self.clock_s,'captures':self.ordinal,
                'last_capture_id':self.last_capture,'configuration':self.config.document(),
                'initial_anchor_scene_m':list(self.anchor),'initial_heading_deg':self.heading,
                'events':self.events.copy(),'training_eligible':False,
                'clock_policy':'discrete virtual events; playback speed never changes acoustic sample rate',
                'persistence':'bounded local memory; download a bundle to retain the current capture'}

class Sessions:
    def __init__(self,store,max_sessions=32,max_captures=128):
        self.store=store;self.sessions=OrderedDict();self.captures=OrderedDict()
        self.max_sessions=max_sessions;self.max_captures=max_captures;self.lock=threading.RLock()
    def create(self,configuration):
        config=from_dict(configuration)
        if config.mode=='replay':self.store.load(config.sample_id) # fail before creating an unknown replay
        session=Session(uuid.uuid4().hex[:20],config,tuple(config.scanner.position_m),config.scanner.heading_deg)
        session.event('configured',mode=config.mode,engine_version=ENGINE_VERSION)
        with self.lock:
            self.sessions[session.identifier]=session
            while len(self.sessions)>self.max_sessions:self.sessions.popitem(last=False)
        return session
    def get(self,identifier):
        try:return self.sessions[identifier]
        except KeyError as exc:raise LookupError('unknown or expired simulation session') from exc
    def captured(self,identifier):
        try:return self.captures[identifier]
        except KeyError as exc:raise LookupError('unknown or expired capture; recapture or use the downloaded bundle') from exc
    def update(self,identifier,configuration):
        new=from_dict(configuration)
        with self.lock:
            session=self.get(identifier)
            if new.mode!=session.config.mode or new.room!=session.config.room or new.sample_id!=session.config.sample_id:
                raise ValueError('create a new session when changing mode, room geometry or replay sample')
            if new!=session.config:
                session.config=new;session.revision+=1;session.state='configured';session.last_capture=None
                session.event('configuration_changed',revision=session.revision)
            return session
    def capture(self,identifier):
        with self.lock:
            session=self.get(identifier);session.state='capturing';session.last_capture=None
            session.event('capture_started',ordinal=session.ordinal,revision=session.revision)
            try:
                result=(virtual_capture(session.config,session.anchor,session.heading,session.ordinal)
                        if session.config.mode=='virtual' else replay_capture(session.config,self.store,session.ordinal))
            except (ValueError,LookupError) as exc:
                session.state='rejected';session.event('capture_rejected',reason=str(exc));raise
            start=session.clock_s;session.event('scan_frame',start+.033,pose_time_s=start)
            count=min(session.config.k,int(result.sample.arrays['rir_valid'].sum()))
            for i in range(count):
                emission=start+.04+i*.25
                session.event('source_emission',emission,pair_index=i,pose_time_s=emission)
                samples=len(result.signals.get(f'pair_{i}_recording',np.zeros(1280)))
                session.event('microphone_capture',emission+samples/16000,pair_index=i,
                              sample_rate_hz=16000,samples=samples,emission_time_s=emission)
            session.clock_s=round(start+.04+max(count,1)*.25,9)
            session.ordinal+=1;session.state='ready';session.last_capture=result.metadata['capture_id']
            session.event('capture_validated',valid_pairs=int(result.sample.arrays['rir_valid'].sum()))
            if session.config.mode=='virtual':
                result.sample.record['observation_payload_ref']=f"/v1/simulation/captures/{session.last_capture}/observation.npz"
                result.sample.record['observation_sensor_type']='virtual_depth_scanner'
            result.metadata['session_revision']=session.revision
            self.captures[session.last_capture]=result;self.captures.move_to_end(session.last_capture)
            while len(self.captures)>self.max_captures:self.captures.popitem(last=False)
            session.history.append({'capture_id':session.last_capture,'configuration':session.config.document(),
                                    'ordinal':session.ordinal-1,'capture_sha256':result.metadata['capture_sha256']})
            session.history=session.history[-32:]
            return result
    def bundle(self,identifier,include_reference=False):
        with self.lock:
            session=self.get(identifier)
            if not session.last_capture:raise ValueError('capture before downloading')
            capture=self.captured(session.last_capture)
            output=io.BytesIO()
            with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
                manifest={'format':'echospace.simulation_bundle.v1','engine_version':ENGINE_VERSION,
                          'session':session.public(),'history':session.history,'capture':capture.metadata,
                          'sample_record':capture.sample.record,'contains_reference':include_reference,
                          'training_eligible':False,'warning':'observation.npz intentionally excludes targets; not a training manifest'}
                z.writestr('session.json',json.dumps(manifest,indent=2,allow_nan=False))
                z.writestr('observation.npz',npz(capture.observation()))
                observed={k:v for k,v in capture.signals.items() if not k.endswith('clean_room_rir')}
                if observed:z.writestr('device_signals.npz',npz(observed))
                if include_reference:
                    z.writestr('REFERENCE_ONLY.json',json.dumps(capture.reference,indent=2,allow_nan=False))
                    clean={k:v for k,v in capture.signals.items() if k.endswith('clean_room_rir')}
                    if clean:z.writestr('REFERENCE_SIGNALS_ONLY.npz',npz(clean))
            return output.getvalue()

def npz(arrays):
    out=io.BytesIO();np.savez_compressed(out,**arrays);return out.getvalue()
