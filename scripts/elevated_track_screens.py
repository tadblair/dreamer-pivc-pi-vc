"""Four inward-tilted wall displays, dimensions in meters."""
import xml.etree.ElementTree as ET
import numpy as np


def add_screens(tree):
    root=tree.getroot(); asset=root.find('asset'); world=root.find('worldbody')
    descriptions={}
    for name,outward in [('east',(1,0,0)),('north',(0,1,0)),
                         ('west',(-1,0,0)),('south',(0,-1,0))]:
        outward=np.array(outward,dtype=float); inward=-outward
        up=(inward+np.array([0.,0.,1.]))/np.sqrt(2)
        normal=(inward-np.array([0.,0.,1.]))/np.sqrt(2)
        right=np.cross(up,normal)
        bottom=outward*1.42+np.array([0.,0.,2.])
        center=bottom+.55/2*up
        brightness=255 if name=='east' else 50
        # Large emission with proportionally small reflectance makes the face
        # lighting-independent to substantially less than one 8-bit gray level.
        value=brightness/(255*1000)
        ET.SubElement(asset,'material',name='screen_'+name,
                      rgba=f'{value} {value} {value} 1',emission='1000',specular='0',shininess='0')
        # Local +Z faces inward and down, toward the low camera on the track.
        ET.SubElement(world,'geom',name='screen_'+name,type='box',
                      pos=' '.join(map(str,center)),xyaxes=' '.join(map(str,np.r_[right,up])),
                      size='.1625 .275 .001',material='screen_'+name)
        descriptions[name]=dict(width_m=.325,height_m=.55,brightness=brightness,
                                bottom_center_m=bottom.tolist(),center_m=center.tolist(),
                                top_center_m=(bottom+.55*up).tolist(),normal=normal.tolist())
    return dict(screens=descriptions,tilt_degrees=45,bottom_height_m=2.,
                bottom_height_above_maze_m=1.,maze_surface_z_m=1.,
                dimensions_convention='width 0.325 m, height along tilted face 0.55 m')


def verify_screens(physics,description):
    for name,s in description['screens'].items():
        gid=physics.model.name2id('screen_'+name,'geom')
        rotation=physics.data.geom_xmat[gid].reshape(3,3)
        center=physics.data.geom_xpos[gid]
        np.testing.assert_allclose(center,s['center_m'],atol=1e-12)
        np.testing.assert_allclose(physics.model.geom_size[gid],[.1625,.275,.001],atol=1e-12)
        np.testing.assert_allclose(center-.275*rotation[:,1],s['bottom_center_m'],atol=1e-12)
        np.testing.assert_allclose((center-.275*rotation[:,1])[2],2.,atol=1e-12)
        np.testing.assert_allclose(center+.275*rotation[:,1],s['top_center_m'],atol=1e-12)
        np.testing.assert_allclose(rotation[:,2],s['normal'],atol=1e-12)
    description['geometry_verified']=True
