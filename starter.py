import bpy
import sys
sys.path.append('.')
from utils import *

def main():
    TEMP_FILE = './temp.pkl'
    PLANE_SCALE_FACTOR = 6

    height, width, num_frame, tcmr_pkl_filepath, ref_video_filepath, gender= read_pkl(TEMP_FILE)

    poses, betas, transs, bboxes, scales, frame_ids = get_param_from_pkl(tcmr_pkl_filepath)

    toggle_viewport(gender)
    bpy.data.scenes['Scene'].frame_end = num_frame

    ar = width/height
    factor = PLANE_SCALE_FACTOR

    add_plane(size =1, scale = (factor, factor/ar, 1))
    add_video_to_plane(ref_video_filepath, num_frame)

    bpy.context.scene.tool_settings.use_keyframe_insert_auto = True
    obj = bpy.data.objects[f'SMPLX-{gender}']
    arm = obj.data
    deselect_all()
    select_object(obj)
    bpy.ops.object.mode_set(mode='POSE')
    obj.select_set(True)

    for n in frame_ids:
        idx = list(frame_ids).index(n)
        if idx == 0:
            obj.keyframe_insert("rotation_quaternion", group="Rotation")
            obj.keyframe_insert("location", group="Location")

        bpy.context.scene.frame_current = n
        pose = poses[idx]
        beta = betas[idx]
        trans = transs[idx]
        scale = scales[idx]

        apply_pose_and_beta(obj, pose, trans, beta)
        bones = bpy.data.objects[f'SMPLX-{gender}'].pose.bones

        for bone in bones:
            if bone.name == 'root':
                bone.location = (trans[0], -trans[2], trans[1])
                bone.keyframe_insert(data_path = 'location', frame = n)

            bone.keyframe_insert(data_path = 'rotation_quaternion', frame = n)
    bpy.context.scene.tool_settings.use_keyframe_insert_auto = False




if __name__ == '__main__':
    main()
