import bpy
import sys
sys.path.append('.')
from utils import *

if __name__ == '__main__':
    height, width, num_frame, tcmr_pkl_filepath, ref_video_filepath, gender= read_pkl('./temp.pkl')
    toggle_viewport(gender)
    bpy.data.scenes['Scene'].frame_end = num_frame

    ar = width/height
    factor = 1

    add_plane(size =1, scale = (factor, factor/ar, 1))
    add_video_to_plane(ref_video_filepath, num_frame)

    poses, betas, transs, bboxes, scales, frame_ids = get_param_from_pkl(tcmr_pkl_filepath)
    bpy.context.scene.tool_settings.use_keyframe_insert_auto = True
    obj = bpy.data.objects[f'SMPLX-{gender}']
    arm = obj.data
    deselect_all()
    select_object(obj)
    bpy.ops.object.mode_set(mode='POSE')
    obj.select_set(True)

    import joblib
    bboxes = joblib.load('./data/to_annotate/abhi/mpt_results.pkl')[1]['bbox']
    plane = bpy.data.objects['Plane']
    pro, all_cx_w, all_cz_w, mesh_height= scale_plane_and_mesh_proportionally(plane, obj, bboxes, width, height)

    masks = glob.glob('./masks/*.png')
    for n in frame_ids:
        idx = list(frame_ids).index(n)
        if idx == 0:
            obj.keyframe_insert("rotation_quaternion", group="Rotation")
            obj.keyframe_insert("location", group="Location")

        bpy.context.scene.frame_current = n
        pose = poses[idx]
        beta = betas[idx]
        trans = transs[idx]
        # scale = [scales[0][idx], scales[1][idx]]
        scale = scales[idx]

        apply_pose_and_beta(obj, pose, trans, beta)
        bones = bpy.data.objects[f'SMPLX-{gender}'].pose.bones

        # scale_plane_acc_to_mask(masks[idx], bpy.data.objects[f'SMPLX-{gender}'], bpy.data.objects['Plane'])
        # plane.keyframe_insert(data_path = 'scale', frame = n)

        for bone in bones:
            # if idx == 0:
            #     if bone.name == 'pelvis':
            #         bone.scale = (1/scale[0], 1/scale[1], 1)
            #         # bone.scale = scale
            #         bone.keyframe_insert(data_path = 'scale', frame = n)

            if bone.name == 'root':
                bone.location = (trans[0], -trans[2], trans[1])
                bone.keyframe_insert(data_path = 'location', frame = n)

            bone.keyframe_insert(data_path = 'rotation_quaternion', frame = n)
    bpy.context.scene.tool_settings.use_keyframe_insert_auto = False

# TODO
 # -  run.sh arparase
 # - scale
 # - bbox, fit
 # - Take the bbox height and sort it in the order
 # Take the bounding box and it's center and then place the root there.
