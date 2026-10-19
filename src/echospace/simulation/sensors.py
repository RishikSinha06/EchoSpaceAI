"""Horizontal depth rays and a declared virtual acoustic device chain."""
import io
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter,sosfilt,chirp,fftconvolve
from .physics import fractional_impulses
from echospace.acoustics.rir import preprocess_wav,RirConfig,TimingEvidence

RATE=16000
SAMPLES=1280

def scan(room,config,frame,rng):
    if not room.inside(config.position_m):raise ValueError('scanner must be strictly inside the room')
    origin=np.asarray(config.position_m,float);free=np.zeros((64,64),bool);wall=free.copy();beams=[];hits=[]
    def cell(point):
        uv=np.floor(frame.scene_to_cell(np.asarray([point]))[0]).astype(int)
        return tuple(uv) if ((uv>=0)&(uv<64)).all() else None
    # endpoint=False for 360 degrees avoids a duplicate ray.
    angles=np.linspace(config.heading_deg-config.fov_deg/2,config.heading_deg+config.fov_deg/2,config.rays,endpoint=config.fov_deg<360)
    for degrees in angles:
        angle=np.deg2rad(degrees);direction=np.array([np.cos(angle),np.sin(angle)])
        distance,returned=room.ray(origin,direction,config.range_m)
        dropped=bool(rng.random()<config.dropout)
        if dropped:
            beams.append({'angle_rad':float(angle),'range_m':None,'returned':False,'dropped':True});continue
        measured=float(np.clip(distance+rng.normal(0,config.range_std_m),.01,config.range_m))
        direction3=np.array([direction[0],0,direction[1]])
        # Infer free space only before the measured endpoint. No truth-mask repair.
        distances=np.arange(0,max(0,measured-.05),.04)
        if len(distances):
            indices=np.floor(frame.scene_to_cell(origin[None,:]+distances[:,None]*direction3)).astype(int)
            indices=indices[((indices>=0)&(indices<64)).all(axis=1)]
            free[indices[:,1],indices[:,0]]=True
        point=origin+measured*direction3;index=cell(point)
        if returned and index:
            wall[index[1],index[0]]=True;hits.append(point)
        beams.append({'angle_rad':float(angle),'range_m':measured,'returned':bool(returned),'dropped':False})
    free[wall]=False
    return {'observed_free':free.astype(np.uint8),'observed_wall':wall.astype(np.uint8),
            'observed_cells':(free|wall).astype(np.uint8),
            'observed_points_scene_m':np.asarray(hits,dtype=np.float32).reshape(-1,3)},beams

def audio(room,pair,config,rng):
    paths=room.path(pair.source_m,pair.receiver_m)
    clean=fractional_impulses(paths,speed=config.speed_m_s)
    if not np.any(clean):raise ValueError('no direct/first-order path arrives in the 80 ms window')
    # Relative device input units; fixed headroom, not calibrated sound pressure.
    clean=clean/max(np.max(np.abs(clean)),1e-12)*.5
    device=clean.copy()
    if config.lowpass_hz is not None:
        device=sosfilt(butter(4,config.lowpass_hz,fs=RATE,output='sos'),device)
    if config.excitation=='sweep':
        time=np.arange(2048)/RATE
        excitation=chirp(time,100,time[-1],6000,method='logarithmic')*np.hanning(len(time))
    else:excitation=np.ones(1)
    signal=fftconvolve(excitation,device)*10**(config.gain_db/20)
    indices=np.arange(len(signal),dtype=float)
    signal=np.interp(indices*(1+config.clock_drift_ppm*1e-6)-config.offset_ms*RATE/1000,
                     indices,signal,left=0,right=0)
    if config.noise_snr_db is not None:
        noise=rng.normal(size=len(signal));rms=np.sqrt(np.mean(signal**2))
        noise=noise/max(np.sqrt(np.mean(noise**2)),1e-12)*rms/10**(config.noise_snr_db/20)
        signal=signal+noise
    clipped=int((np.abs(signal)>config.clip_level).sum())
    signal=np.clip(signal,-config.clip_level,config.clip_level)
    levels=2**(config.adc_bits-1)-1
    recording=np.round(signal/config.clip_level*levels)/levels*config.clip_level
    if config.excitation=='sweep':
        n=1<<int(np.ceil(np.log2(len(recording))))
        x=np.fft.rfft(excitation,n);y=np.fft.rfft(recording,n)
        regularization=1e-7*float(np.max(np.abs(x)**2))
        estimate=np.fft.irfft(y*np.conj(x)/(np.abs(x)**2+regularization),n)[:SAMPLES]
    else:estimate=recording[:SAMPLES]
    buffer=io.BytesIO();wavfile.write(buffer,RATE,np.asarray(estimate,dtype=np.float32))
    processed=preprocess_wav(buffer.getvalue(),RirConfig(),TimingEvidence(
        emission_time_verified=True,upstream_trimmed=False,upstream_normalization='preserve',
        evidence='Known virtual emission reference; declared timing/device faults are intentionally uncompensated'))
    return processed.waveform,{'clean_room_rir':clean.astype(np.float32),'excitation':excitation.astype(np.float32),
                              'recording':recording.astype(np.float32)}, {
        'paths':[dict(p,arrival_ms=p['length_m']/config.speed_m_s*1000) for p in paths],
        'clipped_samples':clipped,'recording_samples':len(recording),
        'processing':processed.metadata,
        'limitations':'relative device units; direct/first-order specular paths only; no scattering/diffraction/higher-order reverb',
        'gain_location':'before ADC and P5 peak normalization; gain may cancel unless clipping/noise/device effects change it',
        'clock_faults_compensated':False}
