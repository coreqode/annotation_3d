import bpy
import pickle
import os
from variables import *

def update_corrective_poseshapes(context):
    if self.smplx_corrective_poseshapes:
        bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')
    else:
        bpy.ops.object.smplx_reset_poseshapes('EXEC_DEFAULT')

def rodrigues_from_pose(armature, bone_name):
    # Use quaternion mode for all bone rotations
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'
    quat = armature.pose.bones[bone_name].rotation_quaternion
    (axis, angle) = quat.to_axis_angle()
    rodrigues = axis
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues

def dump_parameters(seq_name):
    pose = [0.0] * (NUM_SMPLX_JOINTS * 3)
    for index in range(NUM_SMPLX_JOINTS):
        joint_name = SMPLX_JOINT_NAMES[index]
        joint_pose = rodrigues_from_pose(bpy.data.objects['SMPLX-male'], joint_name)
        pose[index*3 + 0] = joint_pose[0]
        pose[index*3 + 1] = joint_pose[1]
        pose[index*3 + 2] = joint_pose[2]
    root = os.getcwd()
    path = f'{root}/data/to_annotate/{seq_name}/annotate/smplx_param.pkl'
    with open(path, 'wb') as fi:
        pickle.dump(pose, fi)

dump_parameters(seq_name = 'abhi')
