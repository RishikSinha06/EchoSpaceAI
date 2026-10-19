"""Strict, bounded configuration independent of API/framework dependencies."""
from dataclasses import dataclass, asdict, fields
import math
import json

SCHEMA='echospace.simulation_config.v1'

def number(value,low,high,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError(f'{name} must be finite within {low}..{high}')

def vector(value,name):
    if not isinstance(value,(list,tuple)) or len(value)!=3:raise ValueError(f'{name} must be [x,y,z] metres')
    for v in value:number(v,-30,30,name)

@dataclass(frozen=True)
class RoomConfig:
    shape: str='rectangle'
    width_m: float=6.
    depth_m: float=5.
    height_m: float=2.8
    reflection: float=.65
    def __post_init__(self):
        if self.shape not in ('rectangle','l_room'):raise ValueError('unknown room shape')
        number(self.width_m,3,10,'width_m');number(self.depth_m,3,10,'depth_m')
        number(self.height_m,2,5,'height_m');number(self.reflection,0,.98,'reflection')
    def vertices(self):
        w,d=self.width_m,self.depth_m
        return [[0.,0.],[w,0.],[w,d],[0.,d]] if self.shape=='rectangle' else [[0.,0.],[w,0.],[w,.55*d],[.55*w,.55*d],[.55*w,d],[0.,d]]

@dataclass(frozen=True)
class ScannerConfig:
    position_m: tuple=(2.,1.1,2.5)
    heading_deg: float=0.
    fov_deg: float=120.
    rays: int=361
    range_m: float=12.
    range_std_m: float=0.
    dropout: float=0.
    def __post_init__(self):
        vector(self.position_m,'scanner position');object.__setattr__(self,'position_m',tuple(self.position_m))
        number(self.heading_deg,-360,360,'heading');number(self.fov_deg,10,360,'fov')
        if type(self.rays)!=int or not 16<=self.rays<=1441:raise ValueError('rays must be 16..1441')
        number(self.range_m,.5,20,'range_m');number(self.range_std_m,0,.2,'range_std_m');number(self.dropout,0,.8,'dropout')

@dataclass(frozen=True)
class PairConfig:
    source_m: tuple=(2.8,1.2,2.5)
    receiver_m: tuple=(3.8,1.2,2.5)
    def __post_init__(self):
        vector(self.source_m,'source');vector(self.receiver_m,'receiver')
        object.__setattr__(self,'source_m',tuple(self.source_m));object.__setattr__(self,'receiver_m',tuple(self.receiver_m))
        if sum((a-b)**2 for a,b in zip(self.source_m,self.receiver_m))<.01:raise ValueError('source and receiver must be at least 0.1 m apart')

@dataclass(frozen=True)
class AudioConfig:
    excitation: str='impulse'
    speed_m_s: float=343.
    noise_snr_db: float | None=None
    gain_db: float=0.
    lowpass_hz: float | None=None
    offset_ms: float=0.
    clock_drift_ppm: float=0.
    adc_bits: int=16
    clip_level: float=1.
    pose_noise_m: float=0.
    def __post_init__(self):
        if self.excitation not in ('impulse','sweep'):raise ValueError('unknown excitation')
        number(self.speed_m_s,300,370,'sound speed')
        if self.noise_snr_db is not None:number(self.noise_snr_db,0,80,'SNR')
        number(self.gain_db,-12,12,'gain');number(self.offset_ms,-2,2,'offset')
        number(self.clock_drift_ppm,-500,500,'clock drift')
        if self.lowpass_hz is not None:number(self.lowpass_hz,500,7900,'lowpass')
        if type(self.adc_bits)!=int or not 8<=self.adc_bits<=24:raise ValueError('ADC bits must be 8..24')
        number(self.clip_level,.05,2,'clip level');number(self.pose_noise_m,0,.1,'pose error')

@dataclass(frozen=True)
class SimulationConfig:
    mode: str='virtual'
    seed: int=0
    sample_id: str | None=None
    k: int=1
    room: RoomConfig=RoomConfig()
    scanner: ScannerConfig=ScannerConfig()
    audio: AudioConfig=AudioConfig()
    pairs: tuple=(PairConfig(),)
    def __post_init__(self):
        if self.mode not in ('virtual','replay'):raise ValueError('mode must be virtual or replay')
        if type(self.seed)!=int or not 0<=self.seed<2**31:raise ValueError('seed must be a nonnegative 31-bit integer')
        if type(self.k)!=int or not 1<=self.k<=8:raise ValueError('K must be 1..8')
        if self.sample_id is not None and (not isinstance(self.sample_id,str) or not 1<=len(self.sample_id)<=256):raise ValueError('invalid sample_id')
        if self.mode=='replay' and not self.sample_id:raise ValueError('replay requires an audited sample_id')
        if not isinstance(self.pairs,(list,tuple)) or not 1<=len(self.pairs)<=8:raise ValueError('provide 1..8 unique pairs')
        ids=[tuple(p.source_m)+tuple(p.receiver_m) for p in self.pairs]
        if len(ids)!=len(set(ids)):raise ValueError('duplicate source/receiver pairs')
        if self.mode=='replay' and (self.audio!=AudioConfig() or self.scanner!=ScannerConfig() or self.room!=RoomConfig() or self.pairs!=(PairConfig(),)):
            raise ValueError('replay is exact: sensor/room fault overrides are not permitted')
    def document(self):return json.loads(json.dumps(asdict(self)))

def from_dict(value):
    if not isinstance(value,dict):raise ValueError('configuration must be an object')
    unknown=set(value)-{f.name for f in fields(SimulationConfig)}
    if unknown:raise ValueError('unknown configuration fields: '+', '.join(sorted(unknown)))
    try:
        data=dict(value)
        for key,cls in (('room',RoomConfig),('scanner',ScannerConfig),('audio',AudioConfig)):
            if key in data:
                if not isinstance(data[key],dict):raise ValueError(f'{key} must be an object')
                data[key]=cls(**data[key])
        if 'pairs' in data:
            if not isinstance(data['pairs'],list) or not 1<=len(data['pairs'])<=8:raise ValueError('provide 1..8 pairs')
            data['pairs']=tuple(PairConfig(**pair) for pair in data['pairs'])
        return SimulationConfig(**data)
    except TypeError as exc:raise ValueError('invalid configuration field/type') from exc
