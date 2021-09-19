import bpy
import json
import os
import glob
import pickle
import sys
import numpy as np
import joblib
# import cv2
from mathutils import Vector, Quaternion
sys.path.append('.')
from variables import *

def toggle_mode_to( mode):
    # bpy.ops.object.posemode_toggle()
    bpy.ops.object.mode_set(mode=mode)

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
    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location = (0, 0, 0), rotation=(1.5707963267949, 0, 0), )
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
    frame_ids = data['frame_ids']
    pelvis = joints_3d[:, 0, :]

    # pred_cam = data['pred_cam']
    # scales = np.zeros((pose.shape[0], 3))
    # scales[:, 0] = pred_cam[:, 0]
    # scales[:, 1] = pred_cam[:, 0]
    # scales[:, 2] = pred_cam[:, 0]

    scales = np.stack([camera_param[:, 0], camera_param[:, 1], camera_param[:, 1]], axis  = -1)

    trans = np.zeros((pose.shape[0], 3))
    trans[:, 0] = camera_param[:, 2]
    trans[:, 1] = camera_param[:,  3]
    return pose, betas, trans, bboxes, scales, frame_ids


def get_bbox(obj):
    bboxes  = np.zeros((8, 3))
    for i, bb in enumerate(obj.bound_box):
        bboxes[i] = obj.matrix_world @ Vector([bb[0],  bb[1], bb[2]])
    return bboxes

def get_world_height(obj):
    bbox = get_bbox(obj)
    uq = np.unique(bbox[..., -1])
    height = np.abs(uq[0]  - uq[1])
    uq_w = np.unique(bbox[..., 0])
    width = np.abs(uq_w[0]  - uq_w[1])
    return height, width

def scale_plane_and_mesh_proportionally(plane, mesh, bboxes, vid_width, vid_height):
    plane_bbox = get_bbox(plane)
    mesh_bbox = get_bbox(mesh)

    uq = np.unique(mesh_bbox[..., -1])
    mesh_height = np.abs(uq[0]  - uq[1])
    uq = np.unique(plane_bbox[..., -1])
    plane_height = np.abs(uq[0]  - uq[1])
    uq_w = np.unique(plane_bbox[..., 0])
    plane_width = np.abs(uq_w[0]  - uq_w[1])

    all_bbox_height = []
    all_cx_w = []
    all_cz_w = []

    for n in range(bboxes.shape[0]):
        bbox = bboxes[n]
        if len(bbox) > 0:
            bbox_cx, bbox_cy, bbox_width, bbox_height = bbox
            all_bbox_height.append(bbox_height)
            cx_o = bbox_cx - vid_width / 2
            cz_o = -(bbox_cy - vid_height)
            p_h = plane_height / vid_height
            p_w = plane_width / vid_width
            cx_w = p_w * cx_o
            cz_w = p_h * cz_o
            all_cx_w.append(cx_w)
            all_cz_w.append(cz_w)

    all_bbox_height.sort(reverse = True)
    bbox_height = np.mean(all_bbox_height[:10])

    propr = bbox_height / vid_height
    rw_height = propr * plane_height
    proportion = rw_height / mesh_height

    return proportion, all_cx_w, all_cz_w, mesh_height

def global_to_local(arm_obj, bone, rot):
    gr = Vector([rot[0], rot[1], rot[2]])
    local_rotation = bone.matrix.inverted() @ arm_obj.matrix_world.inverted() @ Vector(gr)
    return local_rotation

def local_to_global(arm_obj, bone, local_rotation):
   global_rotation = arm_obj.matrix_world @ bone.matrix @ Vector(local_rotation)
   return global_rotation

# def scale_plane_acc_to_mask(mask, mesh, plane):
#     m = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
#     rows, cols = np.where(m > 0)
#     xmin, ymin = np.min(cols), np.min(rows)
#     xmax, ymax = np.max(cols), np.max(rows)
#     bbx_ht = ymax - ymin

#     mesh_height, mesh_width = get_world_height(mesh)
#     plane_height, plane_width = get_world_height(plane)

#     plane_rwh = (mesh_height / bbx_ht) * 1080
#     # toggle_mode_to('OBJECT')
#     bpy.data.objects['Plane'].scale[1] = plane_rwh
#     bpy.data.objects['Plane'].scale[0] = plane_rwh * (plane_width / plane_height)

def get_beta_param(obj, gender, path):
    bpy.ops.object.mode_set(mode='OBJECT')
    regressor_path = os.path.join(path, "data", f"smplx_measurements_to_betas_{gender}.json")

    with open(regressor_path) as f:
        data = json.load(f)
        betas_regressor = ( np.asarray(data["A"]).reshape(-1, 2), np.asarray(data["B"]).reshape(-1, 1),)
        (A, B) = betas_regressor

    # Calculate beta values from measurements
    height_m = bpy.context.window_manager.smplx_tool.smplx_height
    height_cm = height_m * 100.0
    weight_kg = bpy.context.window_manager.smplx_tool.smplx_weight

    v_root = pow(weight_kg, 1.0/3.0)
    measurements = np.asarray([[height_cm], [v_root]])
    betas = A @ measurements + B

    bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')
    return betas
