"""Exact MATLAB outline extruded into convex MuJoCo collision pieces, in meters."""
import json
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET
from dm_control import mujoco
import mujoco as mj

OUT=Path('artifacts/elevated-track');OUT.mkdir(parents=True,exist_ok=True)
outer=np.array([[-.95,1.06],[.95,1.06],[1.1875,1.3],[1.3,1.1875],[1.06,.95],[1.06,-.95],
 [1.3,-1.1875],[1.1875,-1.3],[.95,-1.06],[-.95,-1.06],[-1.1875,-1.3],[-1.3,-1.1875],
 [-1.06,-.95],[-1.06,.95],[-1.3,1.1875],[-1.1875,1.3]])
hole=np.array([[-.94,.94],[-.06,.94],[-.06,.06],[-.94,.06]])
track=Polygon(outer,holes=[hole+s for s in ([0,0],[1,0],[0,-1],[1,-1])])
rim=track.buffer(.02,join_style='mitre').difference(track)
assert track.is_valid and rim.is_valid
assert abs(track.intersection(rim).area)<1e-12
root=ET.Element('mujoco',model='elevated_track')
ET.SubElement(root,'compiler',angle='radian')
ET.SubElement(root,'option',gravity='0 0 -9.81',timestep='.002')
vis=ET.SubElement(root,'visual')
ET.SubElement(vis,'global',offwidth='768',offheight='768')
ET.SubElement(vis,'headlight',ambient='.5 .5 .5',diffuse='.6 .6 .6',specular='0 0 0')
ET.SubElement(vis,'map',znear='.001',zfar='20')
asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
ET.SubElement(asset,'material',name='darkgray',rgba='.22 .22 .22 1',specular='0',shininess='0')
ET.SubElement(asset,'material',name='black',rgba='0 0 0 1',specular='0',shininess='0')
faces=np.array([[0,2,1],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]])
checks={}
for name,poly,z0,z1 in [('track',track,.98,1.),('rim',rim,.98,1.025)]:
    tris=list(shapely.constrained_delaunay_triangles(poly).geoms)
    union=shapely.union_all(tris)
    assert union.symmetric_difference(poly).area<1e-12
    for i,tri in enumerate(tris):
        xy=np.asarray(tri.exterior.coords)[:3]
        if np.linalg.det(np.stack([xy[1]-xy[0],xy[2]-xy[0]]))<0:xy=xy[::-1]
        verts=np.vstack([np.column_stack([xy,np.full(3,z)]) for z in (z0,z1)])
        normal=np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]])
        normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        ident=f'{name}_{i}'
        ET.SubElement(asset,'mesh',name=ident,vertex=' '.join(map(str,verts[faces].ravel())),
            face=' '.join(map(str,np.arange(faces.size))),
            normal=' '.join(map(str,np.repeat(normal,3,axis=0).ravel())))
        ET.SubElement(world,'geom',name=ident,type='mesh',mesh=ident,material='darkgray',friction='.8 .02 .02')
    checks[name]=dict(pieces=len(tris),area_m2=poly.area,bounds=poly.bounds,exact_triangulation=True)
N=2.84;half=N/2;th=.025
for name,pos,size in [('floor',[0,0,-th],[half+2*th,half+2*th,th]),
 ('ceiling',[0,0,N+th],[half+2*th,half+2*th,th]),
 ('west',[-half-th,0,half],[th,half,half]),('east',[half+th,0,half],[th,half,half]),
 ('south',[0,-half-th,half],[half,th,half]),('north',[0,half+th,half],[half,th,half])]:
    ET.SubElement(world,'geom',name=name,type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),material='black')
ET.SubElement(world,'camera',name='overhead',pos='0 0 2.75',xyaxes='1 0 0 0 1 0',fovy='90')
rig=ET.SubElement(world,'body',name='camera_rig',mocap='true',pos='0 0 1.05')
for name,side,yaw in [('left_eye',1,50),('right_eye',-1,-50)]:
    a,p=np.deg2rad([yaw,15])
    forward=np.array([np.cos(a)*np.cos(p),np.sin(a)*np.cos(p),np.sin(p)])
    right=np.array([np.sin(a),-np.cos(a),0.]);up=np.cross(right,forward)
    ET.SubElement(rig,'camera',name=name,pos=f'0 {side*.0065} 0',xyaxes=' '.join(map(str,np.r_[right,up])),fovy='150')
xml=ET.tostring(root,encoding='unicode');(OUT/'scene.xml').write_text(xml)
physics=mujoco.Physics.from_xml_string(xml)
physics.forward()
eyes=[physics.model.name2id(n,'camera') for n in ('left_eye','right_eye')]
np.testing.assert_allclose(np.linalg.norm(np.diff(physics.data.cam_xpos[eyes],axis=0)),.013,atol=1e-12)
np.testing.assert_allclose(physics.data.cam_xpos[eyes,2],1.05,atol=1e-12)
for eye,yaw in zip(eyes,[50,-50]):
    look=-physics.data.cam_xmat[eye].reshape(3,3)[:,2]
    np.testing.assert_allclose(np.rad2deg(np.arctan2(look[1],look[0])),yaw,atol=1e-8)
    np.testing.assert_allclose(np.rad2deg(np.arcsin(look[2])),15,atol=1e-8)
collision_checks=[]
for x,y,expected in [(.011,.023,1.),(.5,.5,0.),(1.,.013,1.),(1.07,.013,1.025),(.07,.5,1.025),
                     (.09,.5,0.),(1.1,0,0.),(.04,.5,1.),(1.2,1.21,1.)]:
    hit=np.array([-1],np.int32)
    distance=mj.mj_ray(physics.model.ptr,physics.data.ptr,np.array([x,y,1.5]),np.array([0.,0.,-1.]),None,True,-1,hit)
    np.testing.assert_allclose(1.5-distance,expected,atol=1e-6,err_msg=f'Ray at {x},{y}')
    collision_checks.append(dict(x=x,y=y,surface_z_m=float(1.5-distance)))
checks.update(room_interior_side_m=N,platform_surface_z_m=1.,slab_thickness_m=.02,
    rim_height_above_surface_m=.025,rim_width_outside_track_m=.02,
    rim_bottom_z_m=.98,camera_height_above_surface_m=.05,eye_separation_m=.013,pitch_up_degrees=15,
    collision_surface_checks=collision_checks,all_geometry_checks_passed=True)
(OUT/'geometry.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
