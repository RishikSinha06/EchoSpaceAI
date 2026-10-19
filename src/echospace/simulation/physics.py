"""Finite polygon-room direct/first-order paths. No higher-order or wave acoustics."""
import numpy as np
from shapely.geometry import Polygon,Point,LineString

class Room:
    def __init__(self,config):
        self.config=config;self.vertices=np.asarray(config.vertices(),dtype=float)
        self.polygon=Polygon(self.vertices)
        self.edges=list(zip(self.vertices,np.roll(self.vertices,-1,axis=0)))
    def inside(self,p):
        return 0<float(p[1])<self.config.height_m and self.polygon.contains(Point(float(p[0]),float(p[2])))
    def visible(self,a,b):
        if not (-1e-7<=a[1]<=self.config.height_m+1e-7 and -1e-7<=b[1]<=self.config.height_m+1e-7):return False
        shape=Point(float(a[0]),float(a[2])) if np.linalg.norm(np.asarray(a)[[0,2]]-np.asarray(b)[[0,2]])<1e-10 else LineString([[a[0],a[2]],[b[0],b[2]]])
        return self.polygon.buffer(1e-8).covers(shape)
    def path(self,source,receiver):
        s,m=np.asarray(source,float),np.asarray(receiver,float)
        if not self.inside(s) or not self.inside(m):raise ValueError('source and receiver must be strictly inside room')
        paths=[]
        def add(kind,points,coefficient):
            length=sum(float(np.linalg.norm(b-a)) for a,b in zip(points,points[1:]))
            paths.append({'kind':kind,'length_m':length,'amplitude':coefficient/max(length,.1),
                          'points_scene_m':[p.tolist() for p in points]})
        if self.visible(s,m):add('direct',[s,m],1.)
        planes=[]
        for i,(a,b) in enumerate(self.edges):
            d=b-a;n=np.array([-d[1],0,d[0]])/np.linalg.norm(d)
            planes.append((f'wall_{i}',n,float(n@np.array([a[0],0,a[1]])),a,b))
        planes.extend([('floor',np.array([0.,1.,0.]),0.,None,None),('ceiling',np.array([0.,1.,0.]),self.config.height_m,None,None)])
        for name,n,offset,a,b in planes:
            ds=float(n@s-offset);dm=float(n@m-offset)
            if ds*dm<=1e-9:continue # same reflecting side only
            image=s-2*ds*n;direction=m-image;den=float(n@direction)
            if abs(den)<1e-10:continue
            fraction=(offset-float(n@image))/den
            if not 0<fraction<1:continue
            hit=image+fraction*direction
            if a is None:
                if not self.polygon.buffer(1e-8).covers(Point(hit[0],hit[2])):continue
            else:
                d=b-a;t=float((hit[[0,2]]-a)@d/(d@d))
                if not -1e-8<=t<=1+1e-8 or not -1e-8<=hit[1]<=self.config.height_m+1e-8:continue
            if not self.visible(s,hit) or not self.visible(hit,m):continue
            add(name,[s,hit,m],self.config.reflection)
        return paths
    def ray(self,origin,direction,maximum):
        p=np.asarray(origin,float)[[0,2]];d=np.asarray(direction,float)
        best=maximum;hit=False
        cross=lambda a,b:float(a[0]*b[1]-a[1]*b[0])
        for a,b in self.edges:
            edge=b-a;den=cross(d,edge)
            if abs(den)<1e-10:continue
            t=cross(a-p,edge)/den;u=cross(a-p,d)/den
            if t>1e-7 and -1e-8<=u<=1+1e-8 and t<=best:
                best=float(t);hit=True
        return best,hit

def fractional_impulses(paths,rate=16000,samples=1280,speed=343.):
    wave=np.zeros(samples,dtype=float)
    for path in paths:
        index=path['length_m']/speed*rate;left=int(np.floor(index));fraction=index-left
        if 0<=left<samples:wave[left]+=path['amplitude']*(1-fraction)
        if 0<=left+1<samples:wave[left+1]+=path['amplitude']*fraction
    return wave
