import os
import math
import time
import json
import random
import open3d
import numpy as np
from gym import spaces
from peg_in_hole_gym.envs.meta_env import MetaEnv
from scipy.spatial.transform import Rotation as R
from utils.perception import CameraIntrinsic, Camera, Frame
from utils import fusion

class PlugIn(MetaEnv):
    action_space=spaces.Box(np.array([-0.8,0,0,-math.pi,-math.pi,-math.pi]),np.array([0.8,0.8,0.8,math.pi,math.pi,math.pi]))
    observation_space = spaces.Box(np.array([-1]), np.array([1]))
    def __init__(self, client, offset=[0,0,0], args=[]):
        self.rest_poses=[-3.126411132142109, -1.3986403400756737, 1.8925739340152319, 2.9060338696988457, -0.013029828104559682, -0.28757678821903976]
        super(PlugIn, self).__init__(client, offset)
        

    def _load_models(self):
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING,1)        
        self.gravity = [0,0,-9.8]
        self.p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
        # ur init
        self.urEndEffectorIndex = 7 # !!!!!
        self.urNumDofs = 6
        ur_base_pose = np.array([0, 0.0, -0.1])+self.offset
        table_base_pose = np.array([0.0,-0.5,-1.3])+self.offset
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | self.p.URDF_USE_INERTIA_FROM_FILE
        self.init_ur(ur_base_pose, self.rest_poses, table_base_pose, flags)
    
        # init charge board
        base_orn = self.p.getQuaternionFromEuler([0,0,0])
        self.objectUid = self.p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/charge_board.urdf"),
                            basePosition=np.array([0.4,0.3,0.5])+self.offset, baseOrientation=base_orn,
                            globalScaling=1)

        # init camera
        self._init_camera_param()
        self.end_axes = DebugAxes(self.p)
        self.camera_axes = DebugAxes(self.p)

    def _reset_internals(self):
        self.vol_bnds = np.array([
            [-2,2],
            [-2,2],
            [-2,2]
        ])
        self.tsdf_vol = fusion.TSDFVolume(self.vol_bnds, voxel_size=0.01)


    def apply_action(self, action):
        action = np.clip(action, self.action_space.low,self.action_space.high)
        self.ur_execute(action, self.urEndEffectorIndex, self.urNumDofs)

    def get_info(self):
        self.tsdf_fusion()
        self.update_debug_axes()
        info = {}
        return 0.,0., self.done, info


    def reset(self, hard_reset=False):
        super().reset(hard_reset=hard_reset)
        self.reset_ur(self.rest_poses)
        self.p.resetJointState(self.objectUid, 1, random.random() * (-math.pi/3))
        # self.p.changeDynamics(self.objectUid, -1, linearDamping=0, angularDamping=0)
        


        obs = []
        return obs

    def render(self, mode='rgb_array'):
        self.pc_visualization()

    def get_end_state(self):
        """Get the position and orientation of the end effector.

        Returns:
        - end_pos: len=3, (x, y, z) in world coordinate system
        - end_orn: len=4, orientation in quaternion representation (x, y, z, w)
        """
        end_state = self.p.getLinkState(self.ur_id, self.urEndEffectorIndex)
        end_pos = end_state[0]
        end_orn = end_state[1]

        return end_pos, end_orn

    def _init_camera_param(self):
        camera_config = "./camera.json"
        with open(camera_config, "r") as j:
            config = json.load(j)
        camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
        # rela_tform = np.array(config["rela_tform"])

        self.camera = Camera(camera_intrinsic, near=0.01, far=5)
        self.camera_intr = camera_intrinsic.K

    def _bind_camera_to_end(self, end_pos, end_orn):
        """设置相机坐标系与末端坐标系的相对位置
        
        Arguments:
        - end_pos: len=3, end effector position
        - end_orn: len=4, end effector orientation, quaternion (x, y, z, w)

        Returns:
        - wcT: shape=(4, 4), transform matrix, represents camera pose in world frame
        """
        relative_offset = [0, 0, 0.1]  # 相机原点相对于末端执行器局部坐标系的偏移量
        end_orn = R.from_quat(end_orn).as_matrix()
        end_x_axis, end_y_axis, end_z_axis = end_orn.T

        wcT = np.eye(4)  # w: world, c: camera, ^w_c T
        wcT[:3, 0] = -end_y_axis  # camera x axis
        wcT[:3, 1] = -end_z_axis  # camera y axis
        wcT[:3, 2] = end_x_axis  # camera z axis
        # wcR = np.array([
        #     [0, 1, 0],
        #     [1, 0, 0],
        #     [0, 0, -1],
        # ])
        # wcT[:3, :3] = wcR.dot(end_orn)
        wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos  # eye position
        return wcT
    
    def tsdf_fusion(self):
        end_pos, end_orn = self.get_end_state()
        # r = R.from_quat(end_orn).as_euler("xyz")
        # print("end_orn", r)
        wcT = self._bind_camera_to_end(end_pos, end_orn)
        cwT = np.linalg.inv(wcT)

        frame = self.camera.render(self.p, cwT)
        assert isinstance(frame, Frame), "Camera render error"

        color_image = frame.color_image()
        depth_im = frame.depth_image()
        # point_cloud = frame.point_cloud()
        # open3d.visualization.draw_geometries([point_cloud])
        # cam_pose = np.eye(4)
        # cam_pose[:3, :3] = R.from_quat(end_orn).as_matrix()
        # cam_pose[:3, 3] = end_pos

        self.tsdf_vol.integrate(color_image, depth_im, self.camera_intr, wcT, obs_weight=1)
    
    def pc_visualization(self):
        raw_pc = self.tsdf_vol.get_point_cloud()
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(raw_pc[:,:3])
        pc.colors = open3d.utility.Vector3dVector(raw_pc[:,3:6]/256)
        open3d.visualization.draw_geometries([pc])

    def update_debug_axes(self):
        # update debug axes and camera position
        end_pos, end_orn = self.get_end_state()
        self.end_axes.update(end_pos, end_orn)

        wcT = self._bind_camera_to_end(end_pos, end_orn)
        self.camera_axes.update(
            pos=wcT[:3, 3],
            orn=R.from_matrix(wcT[:3, :3]).as_quat()
        )

class DebugAxes(object):
    """
    可视化某个局部坐标系, 红色x轴, 绿色y轴, 蓝色z轴
    """
    def __init__(self, client):
        self.uids = [-1, -1, -1]
        self.p = client

    def update(self, pos, orn):
        """
        Arguments:
        - pos: len=3, position in world frame
        - orn: len=4, quaternion (x, y, z, w), world frame
        """
        pos = np.asarray(pos)
        rot3x3 = R.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = self.p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = self.p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = self.p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1], replaceItemUniqueId=self.uids[2])
