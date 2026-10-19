"""Explicit local simulation endpoints; existing audited-sample routes stay strict."""
import io
from typing import Literal
import numpy as np
from fastapi import APIRouter,HTTPException,Response
from pydantic import BaseModel,ConfigDict,Field
from scipy.io import wavfile
from . import ENGINE_VERSION
from .config import SimulationConfig,SCHEMA
from .sessions import Sessions,npz
from echospace.inference.predict import reconstruct
from echospace.inference.registry import RegistryError

class Configuration(BaseModel):
    model_config=ConfigDict(extra='forbid')
    schema_version: Literal['echospace.simulation_config.v1']
    config: dict

class Infer(BaseModel):
    model_config=ConfigDict(extra='forbid',protected_namespaces=())
    model_id: str=Field(min_length=1,max_length=128)
    k: int | None=Field(default=None,ge=1,le=8)
    allow_seen_rooms: bool=False

def router(registry,store,remember,summary,revision,device):
    routes=APIRouter(prefix='/v1/simulation',tags=['simulation'])
    sessions=Sessions(store)
    def attempt(fn):
        try:return fn()
        except LookupError as exc:raise HTTPException(404,str(exc)) from exc
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    @routes.get('/capabilities')
    def capabilities():
        return {'engine_version':ENGINE_VERSION,'configuration_schema':SCHEMA,'defaults':SimulationConfig().document(),
                'modes':['virtual','replay'],'backend':'finite polygon, direct + first-order specular wall/floor/ceiling paths',
                'physical_hardware':False,'training_eligible':False,
                'limits':{'sessions':32,'captures':128,'pairs':8,'rays':1441},
                'limitations':['no higher-order reverb, diffraction, scattering or calibrated device response',
                               'virtual captures are out-of-distribution diagnostics, not held-out validation',
                               'sessions are local memory; download bundles to retain captures']}
    @routes.post('/sessions',status_code=201)
    def create(request:Configuration):return attempt(lambda:sessions.create(request.config).public())
    @routes.get('/sessions/{identifier}')
    def get(identifier:str):return attempt(lambda:sessions.get(identifier).public())
    @routes.patch('/sessions/{identifier}')
    def update(identifier:str,request:Configuration):return attempt(lambda:sessions.update(identifier,request.config).public())
    @routes.post('/sessions/{identifier}/capture')
    def capture(identifier:str):return attempt(lambda:sessions.capture(identifier).public())
    @routes.get('/captures/{capture_id}')
    def captured(capture_id:str):return attempt(lambda:sessions.captured(capture_id).public())
    @routes.get('/captures/{capture_id}/reference.json')
    def reference(capture_id:str):return attempt(lambda:sessions.captured(capture_id).reference)
    @routes.get('/captures/{capture_id}/observation.npz')
    def observation(capture_id:str):
        capture=attempt(lambda:sessions.captured(capture_id))
        return Response(npz(capture.observation()),media_type='application/octet-stream',headers={'Content-Disposition':f'attachment; filename="{capture_id}.observation.npz"'})
    @routes.get('/captures/{capture_id}/signals')
    def signals(capture_id:str,pair:int=0):
        capture=attempt(lambda:sessions.captured(capture_id))
        if not 0<=pair<int(capture.sample.arrays['rir_valid'].sum()):raise HTTPException(422,'invalid pair index')
        data={key:np.asarray(value,dtype=float).tolist() for key,value in capture.signals.items()
              if key.startswith(f'pair_{pair}_') and not key.endswith('clean_room_rir')}
        return {'sample_rate_hz':16000,'observed_signals':data,'clean_reference_included':False}
    @routes.get('/captures/{capture_id}/audio.wav')
    def wav(capture_id:str,pair:int=0):
        capture=attempt(lambda:sessions.captured(capture_id))
        if not 0<=pair<int(capture.sample.arrays['rir_valid'].sum()):raise HTTPException(422,'invalid pair index')
        buffer=io.BytesIO();wavfile.write(buffer,16000,capture.sample.arrays['rir_waveforms'][pair].astype(np.float32))
        return Response(buffer.getvalue(),media_type='audio/wav',headers={'Content-Disposition':f'attachment; filename="{capture_id}.pair{pair}.wav"'})
    @routes.get('/sessions/{identifier}/bundle.zip')
    def bundle(identifier:str,include_reference:bool=False):
        data=attempt(lambda:sessions.bundle(identifier,include_reference))
        return Response(data,media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="{identifier}.simulation.zip"'})
    @routes.post('/sessions/{identifier}/infer')
    def infer(identifier:str,request:Infer):
        with sessions.lock:
            session=attempt(lambda:sessions.get(identifier))
            if not session.last_capture:raise HTTPException(409,'capture the current configuration before inference')
            capture=attempt(lambda:sessions.captured(session.last_capture))
            if not capture.sample.arrays['rir_valid'].any():raise HTTPException(409,'no eligible acoustic pairs: move both devices into observed free space and recapture')
            try:entry=registry.get(request.model_id)
            except RegistryError as exc:raise HTTPException(404,str(exc)) from exc
            split=store.split_of(capture.sample.record['room_id'],entry.fold) if session.config.mode=='replay' else 'synthetic_diagnostic'
            if split in ('train','val') and not request.allow_seen_rooms:raise HTTPException(409,'replay room is seen by this model; choose a held-out sample')
            try:model=registry.load(entry.model_id)
            except RegistryError as exc:raise HTTPException(503,str(exc)) from exc
            result=reconstruct(model,entry,capture.sample,request.k or session.config.k,split=split,revision=revision,device=device)
            result.provenance['simulation']={'engine_version':ENGINE_VERSION,'session_id':identifier,'capture_id':capture.metadata['capture_id'],
                                             'mode':session.config.mode,'physical_hardware':False,'training_eligible':False}
            remember(result);session.state='inferred';session.event('inference_completed',model_id=entry.model_id)
            return summary(result)|{'simulation':result.provenance['simulation'],
                                    'warning':capture.metadata['warning']}
    return routes
