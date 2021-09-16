import bpy
import pickle
import sys
import numpy as np
import joblib
from mathutils import Vector, Quaternion
sys.path.append('.')
from variables import *

def theta_to_rad(angle):
    PI = 22/7
    return angle * (PI / 180)

def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if bone_name == 'pelvis':
        rod = Vector((np.pi, 0, 0))
        angle_rad = rod.length
        axis = rod.normalized()
        quat2 = Quaternion(axis, angle_rad)

        ## First you want to rotate around the pelvis
        quat = quat2 @ quat


    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        # SMPL-X is adding the reference rodrigues rotation to the
        # relaxed hand rodrigues rotation, so we have to do the same here.
        # This means that pose values for relaxed hand model cannot be
        # interpreted as rotations in the local joint coordinate system of the relaxed hand.
        # https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/body_models.py#L1190
        #   full_pose += self.pose_mean
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result

        """
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        angle_rad_reference = rod_reference.length
        axis_reference = rod_reference.normalized()
        quat_reference = Quaternion(axis_reference, angle_rad_reference)
        # Rotate first into reference pose and then add the target pose
        armature.pose.bones[bone_name].rotation_quaternion = quat_reference @ quat
        """

def add_plane(size, scale):
    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location = (0, 1, 0), rotation=(1.5707963267949, 0, 0), )
    bpy.context.object.scale[0] = scale[0]
    bpy.context.object.scale[1] = scale[1]
    bpy.context.object.scale[2] = scale[2]
    bpy.context.view_layer.update()
    width = bpy.data.objects['Plane'].dimensions.x
    height = bpy.data.objects['Plane'].dimensions.y
    bpy.context.object.location[2] = height/2

def apply_pose_and_beta(obj,  body_pose, transl, beta):
    if obj.type == 'MESH':
        armature = obj.parent
    else:
        armature = obj
        obj = armature.children[0]
        bpy.context.view_layer.objects.active = obj  # mesh needs to be active object for recalculating joint locations

    global_orient = np.array([body_pose[0] , body_pose[1] ,  body_pose[2] ])
    body_pose = np.array(body_pose)[3:-6]

    if body_pose.shape[0] != (NUM_SMPLX_BODYJOINTS * 3):
        raise ValueError(f"Invalid body pose dimensions: {body_pose.shape}")

    body_pose = np.array(body_pose).reshape(NUM_SMPLX_BODYJOINTS, 3)

    jaw_pose = np.zeros((3))
    left_hand_pose = np.zeros((1, 3))
    rigth_hand_pose = np.zeros((1, 3))
    betas = np.array(beta).reshape(-1).tolist()
    expression = np.zeros((17)).tolist()

    set_pose_from_rodrigues(armature, "pelvis", global_orient)

    # global_orient = np.array([np.pi, 0, 0])
    # set_pose_from_rodrigues(armature, "pelvis", global_orient)

    for index in range(NUM_SMPLX_BODYJOINTS):
        pose_rodrigues = body_pose[index]
        bone_name = SMPLX_JOINT_NAMES[index + 1]  # body pose starts with left_hip
        set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

    set_pose_from_rodrigues(armature, "jaw", jaw_pose)

    # Left hand
    # start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3
    # for i in range(0, NUM_SMPLX_HANDJOINTS):
    #     pose_rodrigues = left_hand_pose[i]
    #     bone_name = SMPLX_JOINT_NAMES[start_name_index + i]
    #     pose_relaxed_rodrigues = self.hand_pose_relaxed[i]
    #     set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)

    # # Right hand
    # start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3 + NUM_SMPLX_HANDJOINTS
    # for i in range(0, NUM_SMPLX_HANDJOINTS):
    #     pose_rodrigues = right_hand_pose[i]
    #     bone_name = SMPLX_JOINT_NAMES[start_name_index + i]
    #     pose_relaxed_rodrigues = self.hand_pose_relaxed[NUM_SMPLX_HANDJOINTS + i]
    #     set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)


def deselect_all():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None

def select_object(obj):
    bpy.context.view_layer.objects.active = obj

def add_video_to_plane(filepath, num_frames):
    all_mat = bpy.data.materials
    mat = all_mat.get('Material')
    if mat is None:
        bpy.data.materials.new('Material')
        mat = bpy.data.materials['Material']
        mat.use_nodes = True

    obj = bpy.context.active_object
    if len(obj.material_slots) == 0:
        bpy.ops.object.material_slot_add()

    obj.material_slots[0].material = mat

    nt = mat.node_tree
    mat_out  = nt.nodes['Material Output']
    emission = nt.nodes.new('ShaderNodeEmission')
    image_texture = nt.nodes.new('ShaderNodeTexImage')
    image_texture.image_user.use_auto_refresh = True
    nt.links.new(emission.outputs['Emission'], mat_out.inputs['Surface'])
    nt.links.new(image_texture.outputs['Color'], emission.inputs['Color'])

    image_texture.image = bpy.data.images.load(filepath)
    image_texture.image_user.frame_duration = num_frames

def toggle_viewport(gender):
    bpy.data.objects[f'SMPLX-{gender}'].hide_set(False)
    bpy.data.objects[f'SMPLX-mesh-{gender}'].hide_set(False)

def read_pkl(filepath):
    with open(filepath, 'rb') as fi:
        d = pickle.load(fi)
    height = d['height']
    width = d['width']
    num_frame = d['num_frame']
    tcmr_pkl_filepath = d['tcmr_pkl_filepath']
    ref_video_filepath = d['ref_video_filepath']
    gender = d['gender']
    return height, width, num_frame, tcmr_pkl_filepath, ref_video_filepath, gender

def get_param_from_pkl(filepath, jb = True):
    if jb:
        data = joblib.load(filepath)
    data = data[1]
    bboxes = data['bboxes']
    pose = data['pose']
    betas = data['betas']
    joints_3d = data['joints3d']
    camera_param = data['orig_cam']
    pelvis = joints_3d[:, 0, :]

    pred_cam = data['pred_cam']
    scales = np.zeros((pose.shape[0], 3))
    scales[:, 0] = pred_cam[:, 0]
    scales[:, 1] = pred_cam[:, 0]
    scales[:, 2] = pred_cam[:, 0]

    trans = np.zeros((pose.shape[0], 3))
    trans[:, 0] = camera_param[:, 2]
    trans[:, 1] = camera_param[:,  3]
    return pose, betas, trans, bboxes, scales


def get_bbox(obj):
    bboxes  = np.zeros((8, 3))
    for i, bb in enumerate(obj.bound_box):
        bboxes[i] = obj.matrix_world @ Vector([bb[0],  bb[1], bb[2]])
    return bboxes

if __name__ == '__main__':
    height, width, num_frame, tcmr_pkl_filepath, ref_video_filepath, gender= read_pkl('./temp.pkl')
    toggle_viewport(gender)
    bpy.data.scenes['Scene'].frame_end = num_frame

    ar = width/height
    factor = 6
    add_plane(size =1, scale = (factor, factor/ar, 1))
    add_video_to_plane(ref_video_filepath, num_frame)

    poses, betas, transs, bboxes, scales = get_param_from_pkl(tcmr_pkl_filepath)
    bpy.context.scene.tool_settings.use_keyframe_insert_auto = True
    obj = bpy.data.objects[f'SMPLX-{gender}']
    arm = obj.data
    deselect_all()
    select_object(obj)
    bpy.ops.object.mode_set(mode='POSE')
    obj.select_set(True)

    plane_bbox = get_bbox(bpy.data.objects['Plane'])
    mesh_bbox = get_bbox(obj)
    uq = np.unique(mesh_bbox[..., -1])
    mesh_height = np.abs(uq[0]  - uq[1])
    uq = np.unique(plane_bbox[..., -1])
    plane_height = np.abs(uq[0]  - uq[1])


    ### Del later
    # import pickle
    # with open('./abhi2.pkl', 'rb') as fi:
    #     bboxes = pickle.load(fi)

    # all_bbox_height = []
    # for n in range(poses.shape[0]):
    #     bbox = bboxes[n]
    #     if len(bbox) > 0:
    #         xmin, ymin, xmax, ymax, _ = bbox[0]
    #     bbox_width = xmax - xmin
    #     all_bbox_height.append(ymax - ymin)
    # all_bbox_height.sort(reverse = True)
    # bbox_height = np.mean(all_bbox_height[:5])

    for n in range(poses.shape[0]):
        if n == 0:
            obj.keyframe_insert("rotation_quaternion", group="Rotation")
            obj.keyframe_insert("location", group="Location")
        pose = poses[n]
        beta = betas[n]
        trans = transs[n]
        scale = scales[n]
        # propr = bbox_height / height
        # rw_height = propr * plane_height

        # pro = rw_height / mesh_height

        apply_pose_and_beta(obj, pose, trans, beta)
        bpy.context.scene.frame_current = n
        bones = bpy.data.objects[f'SMPLX-{gender}'].pose.bones
        # bpy.data.objects['Plane'].scale = (scale[0], scale[1], 1)

        for bone in bones:
            if bone.name == 'root':
                bone.location = (trans[0], -trans[2], trans[1])
                # bone.scale = (scale[0], scale[1], scale[2])
                # bone.scale = (pro, pro, pro)
                bone.keyframe_insert(data_path = 'location', frame = n)
            bone.keyframe_insert(data_path = 'rotation_quaternion', frame = n)

    bpy.context.scene.tool_settings.use_keyframe_insert_auto = False









# TODO
 # -  run.sh arparase
 # -  animation time (end)
 # - scale
 # - bbox, fit
 # - body texture rainbow
 # - Take the bbox height and sort it in the order
 # Take the bounding box and it's center and then place the root there.
