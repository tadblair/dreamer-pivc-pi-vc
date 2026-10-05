"""61 head-oriented proximity rays; visualization only, no model observations."""
import numpy as np
import mujoco
from PIL import ImageDraw

CONTACT = .2
RANGE = 1.2
ANGLES = np.asarray([(0, 0)] + [(plane, tilt) for plane in range(0, 180, 30)
    for tilt in range(-150, 151, 30) if tilt != 0], dtype=np.float64)

def ray_directions(forward):
    forward = np.asarray(forward, np.float64)[:2]
    forward = forward / np.linalg.norm(forward)
    front = np.r_[forward, 0.]
    left = np.array([-front[1], front[0], 0.])
    plane, tilt = np.deg2rad(ANGLES).T
    horizontal = np.cos(plane)[:, None] * front + np.sin(plane)[:, None] * left
    return np.ascontiguousarray(np.sin(tilt)[:, None] * horizontal - np.cos(tilt)[:, None] * [0., 0., 1.])

def activation(distance):
    distance = np.asarray(distance, np.float64)
    return np.where(distance >= 0, np.clip((RANGE - distance) / (RANGE - CONTACT), 0, 1), 0.)

def cast_rays(model, data, origin, directions, excluded_geoms, collision_type=1, collision_affinity=1):
    """Nearest external geometry capable of colliding with the agent.

    mj_ray's single body exclusion cannot exclude the entire articulated agent.
    Temporarily select collision geoms through group masks and visibility,
    restoring them before returning. Memory Maze's physical floor is transparent
    (separate non-colliding textured geoms draw it), but must still be sensed.
    Physics state and collision masks are untouched.
    """
    eligible = (((model.geom_contype & collision_affinity) != 0) |
                ((model.geom_conaffinity & collision_type) != 0))
    eligible[np.asarray(excluded_geoms, dtype=int)] = False
    original = model.geom_group.copy()
    original_alpha = model.geom_rgba[:, 3].copy()
    original_material_alpha = model.mat_rgba[:, 3].copy()
    distances = np.empty(len(directions), np.float64)
    hits = np.empty(len(directions), np.int32)
    group = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    try:
        model.geom_group[:] = np.where(eligible, 0, 5)
        model.geom_rgba[eligible, 3] = 1
        materials = np.unique(model.geom_matid[eligible])
        model.mat_rgba[materials[materials >= 0], 3] = 1
        for i, direction in enumerate(directions):
            hit = np.full(1, -1, np.int32)
            distances[i] = mujoco.mj_ray(model, data, np.asarray(origin, np.float64), direction,
                                          group, True, -1, hit)
            hits[i] = hit[0]
    finally:
        model.geom_group[:] = original
        model.geom_rgba[:, 3] = original_alpha
        model.mat_rgba[:, 3] = original_material_alpha
    assert not np.isin(hits[hits >= 0], excluded_geoms).any()
    return distances, activation(distances), hits

class RaySensors:
    def __init__(self, physics):
        self.physics = physics
        self.model, self.data = physics.model.ptr, physics.data.ptr
        self.shell = physics.model.name2id('walker/shell', 'geom')
        self.excluded = [i for i in range(self.model.ngeom)
            if (physics.model.id2name(i, 'geom') or '').startswith('walker/')]
        assert self.shell in self.excluded

    def read(self, forward):
        origin = self.data.geom_xpos[self.shell].copy()
        directions = ray_directions(forward)
        distances, values, hits = cast_rays(self.model, self.data, origin, directions, self.excluded,
            int(self.model.geom_contype[self.shell]), int(self.model.geom_conaffinity[self.shell]))
        return dict(origin=origin, directions=directions, distance=distances, activity=values, geom_id=hits)

def sensor_locations():
    plane, tilt = ANGLES.T
    azimuth = np.deg2rad(plane + np.where(tilt < 0, 180, 0))
    radius = np.abs(tilt) / 30 * 55
    return np.column_stack([384 - radius * np.sin(azimuth), 375 - radius * np.cos(azimuth)])

def draw_sensor_panels(canvas, readings, fonts):
    """Lower-left angular map, lower-right legend; each panel is 768 square."""
    draw = ImageDraw.Draw(canvas)
    y0 = 768
    draw.line([(16, y0), (1520, y0)], fill='#344457', width=2)
    draw.line([(768, y0 + 16), (768, 1520)], fill='#344457', width=2)
    draw.text((64, y0 + 22), 'PROXIMITY  /  61 sensor rays', font=fonts['title'], fill='#eef6ff')
    for angle in range(0, 360, 30):
        a = np.deg2rad(angle)
        draw.line([(384, y0 + 375), (384 - 275*np.sin(a), y0 + 375 - 275*np.cos(a))], fill='#223247')
    for r in [55, 110, 165, 220, 275]:
        draw.ellipse((384-r, y0+375-r, 384+r, y0+375+r), outline='#34465b', width=1)
    for (x, y), value in zip(sensor_locations(), readings['activity']):
        shade = int(round(255 * value))
        draw.ellipse((x-11, y0+y-11, x+11, y0+y+11), fill=(shade, shade, shade), outline='#8d9daf', width=1)
    draw.text((384, y0+75), 'FRONT', anchor='mm', font=fonts['small'], fill='#b3c5d9')
    draw.text((384, y0+675), 'BACK', anchor='mm', font=fonts['small'], fill='#b3c5d9')
    draw.text((58, y0+375), 'LEFT', anchor='mm', font=fonts['small'], fill='#b3c5d9')
    draw.text((712, y0+375), 'RIGHT', anchor='mm', font=fonts['small'], fill='#b3c5d9')
    draw.text((64, y0+706), 'Center: down  ·  Rings: 30°, 60°, 90°, 120°, 150°', font=fonts['small'], fill='#b3c5d9')
    draw.text((64, y0+733), 'Directions follow the head; sphere rolling does not rotate the array.', font=fonts['small'], fill='#b3c5d9')
    x0 = 832
    draw.text((x0, y0+22), 'SENSOR KEY', font=fonts['title'], fill='#eef6ff')
    draw.text((x0, y0+89), 'White: surface at body contact (0.2 units)', font=fonts['label'], fill='#eef6ff')
    draw.text((x0, y0+124), 'Black: no surface within 1.2 units', font=fonts['label'], fill='#eef6ff')
    for j in range(561):
        v = round(j / 560 * 255)
        draw.line([(x0+j, y0+183), (x0+j, y0+225)], fill=(v, v, v))
    draw.rectangle((x0, y0+183, x0+560, y0+225), outline='#8d9daf', width=1)
    draw.text((x0, y0+240), '0  inactive', font=fonts['small'], fill='#b3c5d9')
    draw.text((x0+560, y0+240), '1  fully active', anchor='ra', font=fonts['small'], fill='#b3c5d9')
    # Distance-response curve with clearly marked contact and range limits.
    ox, oy, width, height = x0+48, y0+475, 492, 145
    draw.line([(ox, oy-height), (ox, oy), (ox+width, oy)], fill='#8d9daf', width=2)
    xy = lambda d, a: (ox + width*d/1.4, oy-height*a)
    draw.line([xy(0, 1), xy(.2, 1), xy(1.2, 0), xy(1.4, 0)], fill='#eef6ff', width=3)
    for d in (.2, .7, 1.2):
        xx, yy = xy(d, 0)
        draw.line([(xx, yy), (xx, yy+5)], fill='#8d9daf')
        draw.text((xx, yy+12), f'{d:.1f}', anchor='ma', font=fonts['small'], fill='#b3c5d9')
    draw.text((ox-15, oy-height), '1', anchor='rm', font=fonts['small'], fill='#b3c5d9')
    draw.text((ox-15, oy), '0', anchor='rm', font=fonts['small'], fill='#b3c5d9')
    draw.text((ox+width/2, oy+47), 'Distance from sphere center (simulation units)', anchor='ma', font=fonts['small'], fill='#b3c5d9')
    draw.text((x0, y0+575), 'Ray origin: center of the rolling sphere', font=fonts['label'], fill='#eef6ff')
    draw.text((x0, y0+612), 'Floor and solid obstacles are sensed; the agent is excluded.', font=fonts['small'], fill='#b3c5d9')
    draw.text((x0, y0+644), 'The downward sensor is normally white on the floor.', font=fonts['small'], fill='#b3c5d9')
    draw.text((x0, y0+710), 'Sensor visualization only · policy still uses vision', font=fonts['small'], fill='#50e3ee')
