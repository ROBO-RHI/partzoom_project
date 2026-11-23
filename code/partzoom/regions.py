"""Oracle part regions for RLBench tasks (privileged simulator information).

Each task defines two regions, both as world-frame axis-aligned boxes:
  region 0 = manipulated part (what the gripper holds / pushes)
  region 1 = receiving part (where it must go)

Two granularities:
  "part"   : the specific task-relevant part (e.g. the chosen spoke of the peg base)
  "object" : the whole object that contains the part (e.g. the entire peg base
             with all three spokes), used for the whole-object-zoom comparison.

Boxes are read from the live simulator at every observation, so they track the
current (visible) pose of the objects. Future demonstration actions are never used.
"""
import numpy as np

from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.const import ObjectType

SHAPE_NAMES = ['cube', 'cylinder', 'triangular_prism', 'star', 'moon']
TARGET_COLORS = ['green', 'blue', 'pink', 'yellow']
NUM_REGIONS = 2


def _obj_aabb(obj):
    """World AABB (2, 3) of one object's own bounding box."""
    bb = np.array(obj.get_bounding_box()).reshape(3, 2)  # [[minx,maxx],...]
    corners = np.array(np.meshgrid(bb[0], bb[1], bb[2])).reshape(3, -1).T
    m = obj.get_matrix()
    world = corners @ m[:3, :3].T + m[:3, 3]
    return np.stack([world.min(0), world.max(0)])


def _tree_aabb(obj):
    """World AABB of an object and all shapes below it."""
    boxes = [_obj_aabb(obj)] if obj.get_type() == ObjectType.SHAPE else []
    for s in obj.get_objects_in_tree(object_type=ObjectType.SHAPE):
        boxes.append(_obj_aabb(s))
    boxes = np.stack(boxes)
    return np.stack([boxes[:, 0].min(0), boxes[:, 1].max(0)])


def _point_box(pos, half):
    pos = np.asarray(pos)
    half = np.asarray(half)
    return np.stack([pos - half, pos + half])


def _get(name):
    return Object.get_object(name)


def _chosen_pillar():
    centre = np.array(_get('success_centre').get_position())
    pillars = [Shape('pillar%d' % i) for i in range(3)]
    d = [np.linalg.norm(np.array(p.get_position())[:2] - centre[:2])
         for p in pillars]
    return pillars[int(np.argmin(d))]


def get_regions(task_name, variation, granularity='part'):
    """Return float32 array (NUM_REGIONS, 2, 3) of world AABBs [min, max]."""
    if task_name == 'insert_onto_square_peg':
        ring = _obj_aabb(_get('square_ring'))
        pillar = _chosen_pillar()
        if granularity == 'part':
            recv = _obj_aabb(pillar)
        else:  # whole peg base with all spokes
            recv = _tree_aabb(pillar.get_parent())
        regions = [ring, recv]
    elif task_name == 'place_shape_in_shape_sorter':
        shape = _get(SHAPE_NAMES[variation])
        man = _obj_aabb(shape)
        if granularity == 'part':
            # slot = drop point of this shape, box sized like the shape footprint
            dp = np.array(_get('%s_drop_point' % SHAPE_NAMES[variation])
                          .get_position())
            half = (man[1] - man[0]) / 2
            half = np.array([max(half[0], half[1])] * 2 + [0.03]) + 0.01
            recv = _point_box(dp, half)
        else:
            recv = _tree_aabb(_get('shape_sorter'))
        regions = [man, recv]
    elif task_name == 'slide_block_to_color_target':
        man = _obj_aabb(_get('block'))
        sensor = _get('success%d' % (variation + 1))
        if granularity == 'part':
            pos = np.array(sensor.get_position())
            recv = _point_box(pos, [0.06, 0.06, 0.03])
        else:  # all four targets together
            pos = np.stack([np.array(_get('success%d' % (i + 1)).get_position())
                            for i in range(4)])
            recv = np.stack([pos.min(0) - [0.06, 0.06, 0.03],
                             pos.max(0) + [0.06, 0.06, 0.03]])
        regions = [man, recv]
    else:
        raise NotImplementedError(task_name)
    return np.stack(regions).astype(np.float32)


def install_region_hook(env, task_name_getter, variation_getter):
    """Wrap Scene.get_observation so every Observation carries region boxes
    in obs.misc['part_regions'] and obs.misc['object_regions']."""
    scene = env._scene
    orig = scene.get_observation

    def get_observation_with_regions():
        obs = orig()
        task_name = task_name_getter()
        if task_name is not None:
            var = variation_getter()
            obs.misc['part_regions'] = get_regions(task_name, var, 'part')
            obs.misc['object_regions'] = get_regions(task_name, var, 'object')
        return obs

    scene.get_observation = get_observation_with_regions
    return orig
