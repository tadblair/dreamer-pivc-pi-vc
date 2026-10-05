"""Named kinematic plexiglass barriers for the elevated-track scene, SI units."""
import xml.etree.ElementTree as ET
import numpy as np

# Current canonical positions: original coordinates rotated 180 degrees about
# (0,0). Labels and raised-panel memberships are unchanged; H/V is invariant.
BARRIERS={
 'B1':(.195,-1.,'V'),'B2':(-.195,-1.,'V'),'B3':(0.,-.805,'H'),
 'B4':(1.,-.195,'H'),'B5':(.805,0.,'V'),'B6':(1.,.195,'H'),
 'B7':(0.,-.195,'H'),'B8':(-.195,0.,'V'),'B9':(0.,.195,'H'),
 'B10':(.195,0.,'V'),'B11':(-1.,-.195,'H'),'B12':(-1.,.195,'H'),
 'B13':(-.805,0.,'V'),'B14':(0.,.805,'H'),'B15':(-.195,1.,'V'),'B16':(.195,1.,'V')}
CONFIGS={0:[1,2,7,9,15,16],1:[1,2,9,15,16],2:[1,2,7,15,16],
         3:[1,2,6,7,9,15],4:[5,13,14],5:[2,5,6,7,9,15,16],6:[1,5,7,9,11,13,15,16]}
HEIGHT=.8128;THICKNESS=.005;SURFACE=1.

def add_barriers(tree,config,width=.30,dynamic=False):
    raised={f'B{i}' for i in CONFIGS[config]}
    root=tree.getroot();asset=root.find('asset');world=root.find('worldbody')
    ET.SubElement(asset,'material',name='plexiglass',rgba='.9 .9 .9 .075',specular='.8',shininess='.9')
    ET.SubElement(asset,'material',name='plexiglass_edge',rgba='.8 .8 .8 .30',specular='.5',shininess='.7')
    description={}
    for name,(x,y,orientation) in BARRIERS.items():
        up=name in raised;z=SURFACE if up else SURFACE-HEIGHT/2
        body=ET.SubElement(world,'body',name=name,mocap='true',pos=f'{x} {y} {z}')
        # H runs along X; V runs along Y. The thickness is orthogonal to that span.
        hx,hy=(width/2,THICKNESS/2) if orientation=='H' else (THICKNESS/2,width/2)
        ET.SubElement(body,'geom',name=name+'_panel',type='box',size=f'{hx} {hy} {HEIGHT/2}',material='plexiglass')
        # Submillimeter edge highlights aid perception without opaque support frames.
        for end in (-1,1):
            sx,sy=(.0003,hy) if orientation=='H' else (hx,.0003)
            ex,ey=(end*hx,0) if orientation=='H' else (0,end*hy)
            ET.SubElement(body,'geom',name=f'{name}_vertical_{end}',type='box',pos=f'{ex} {ey} 0',
                          size=f'{sx} {sy} {HEIGHT/2}',material='plexiglass_edge',contype='0',conaffinity='0')
            # Do not overlay an edge accent coplanar with the platform when down.
            if up or end==-1 or dynamic:
                ET.SubElement(body,'geom',name=f'{name}_horizontal_{end}',type='box',pos=f'0 0 {end*HEIGHT/2}',
                              size=f'{hx} {hy} .0003',material='plexiglass_edge',contype='0',conaffinity='0')
        description[name]=dict(x_m=x,y_m=y,orientation=orientation,state='up' if up else 'down',
            center_z_m=z,bottom_z_m=z-HEIGHT/2,top_z_m=z+HEIGHT/2,width_m=width,
            thickness_m=THICKNESS,height_m=HEIGHT)
    return dict(configuration=config,raised=sorted(raised,key=lambda n:int(n[1:])),barriers=description,
        layout='rotated_180_degrees_labels_preserved',
        panel_opacity=.075,edge_opacity=.30,optics='Rasterized translucency and specular highlights; no physical refraction',
        positioning='Kinematic bodies; static configuration during the recorded camera replay')

def verify_barriers(physics,description):
    for name,b in description['barriers'].items():
        i=physics.model.name2id(name+'_panel','geom')
        np.testing.assert_allclose(physics.data.geom_xpos[i],[b['x_m'],b['y_m'],b['center_z_m']],atol=1e-12)
        expected=[b['width_m']/2,THICKNESS/2,HEIGHT/2]
        if b['orientation']=='V':expected[:2]=expected[:2][::-1]
        np.testing.assert_allclose(physics.model.geom_size[i],expected,atol=1e-12)
        np.testing.assert_allclose(b['top_z_m'],1.4064 if b['state']=='up' else 1.,atol=1e-12)
        np.testing.assert_allclose(b['bottom_z_m'],.5936 if b['state']=='up' else .1872,atol=1e-12)
    description['geometry_verified']=True


def set_barrier_configuration(physics,description,config):
    """Instantaneous configuration change for trajectory replay; no physics step."""
    raised={f'B{i}' for i in CONFIGS[config]}
    for name,b in description['barriers'].items():
        up=name in raised;z=SURFACE if up else SURFACE-HEIGHT/2
        body=physics.model.name2id(name,'body')
        physics.data.mocap_pos[physics.model.body_mocapid[body]]=[b['x_m'],b['y_m'],z]
        edge=physics.model.name2id(name+'_horizontal_1','geom')
        physics.model.geom_rgba[edge]=[.8,.8,.8,.30 if up else 0]
        b.update(state='up' if up else 'down',center_z_m=z,bottom_z_m=z-HEIGHT/2,top_z_m=z+HEIGHT/2)
    description.update(configuration=config,raised=sorted(raised,key=lambda n:int(n[1:])))
    physics.forward();verify_barriers(physics,description)
