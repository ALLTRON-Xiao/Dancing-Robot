
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1Robot(LeggedRobot):
    def _create_envs(self):
        super()._create_envs()
        
        # find all hip dofs, obeying order in self.dof_names
        self.hip_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.hip_dof_name]):
                self.hip_dof_indices.append(i)
        self.hip_dof_indices = torch.tensor(self.hip_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
        self.arm_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.arm_dof_name]):
                self.arm_dof_indices.append(i)
        self.arm_dof_indices = torch.tensor(self.arm_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
        self.waist_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.waist_dof_name]):
                self.waist_dof_indices.append(i)
        self.waist_dof_indices = torch.tensor(self.waist_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)

        self.ankle_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.ankle_dof_name]):
                self.ankle_dof_indices.append(i)
        self.ankle_dof_indices = torch.tensor(self.ankle_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)

        self.hip_knee_dof_indices = []
        for i in range(len(self.dof_names)):
            if any([s in self.dof_names[i] for s in self.cfg.asset.hip_knee_dof_name]):
                self.hip_knee_dof_indices.append(i)
        self.hip_knee_dof_indices = torch.tensor(self.hip_knee_dof_indices, dtype=torch.long, device=self.device, requires_grad=False)

        #//mine
        self.contact_duration = torch.zeros(
            self.num_envs, 
            device=self.device,
            dtype=torch.float
        )
        self.stable_steps_counter = torch.zeros(
            self.num_envs, 
            device=self.device,
            dtype=torch.long  # 整数类型（用于计数）
        )
        self.elapsed_time = torch.zeros(
            self.num_envs, 
            device=self.device,
            dtype=torch.float  # 浮点类型（用于记录时间）
        )
        
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    

    ##///
    def _reward_ankle_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos[:, self.ankle_dof_indices] - self.dof_pos_limits[self.ankle_dof_indices, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos[:, self.ankle_dof_indices] - self.dof_pos_limits[self.ankle_dof_indices, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    def _reward_arm_dof_deviation(self):
        arm_dof_err = torch.sum(torch.abs(self.dof_pos[:, self.arm_dof_indices] - self.default_dof_pos[:, self.arm_dof_indices]), dim=1)
        return arm_dof_err
    
    def _reward_waist_dof_deviation(self):
        waist_dof_err = torch.sum(torch.abs(self.dof_pos[:, self.waist_dof_indices] - self.default_dof_pos[:, self.waist_dof_indices]), dim=1)
        return waist_dof_err
    
    def _reward_hip_dof_deviation(self):
        hip_dof_err = torch.sum(torch.abs(self.dof_pos[:, self.hip_dof_indices] - self.default_dof_pos[:, self.hip_dof_indices]), dim=1)
        return hip_dof_err
    
    def _reward_hip_knee_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel[:, self.hip_knee_dof_indices] - self.dof_vel[:, self.hip_knee_dof_indices]) / self.dt), dim=1)

    def _reward_hip_knee_dof_torques(self):
        return torch.sum(torch.square(self.torques[:, self.hip_knee_dof_indices]), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    def _reward_feet_slip(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        # contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        # penalize = torch.square(contact_feet_vel[:, :, :2])
        # return torch.sum(penalize, dim=(1,2))
        contact_vel_xy = torch.norm(self.feet_vel[:, :, :2], dim=2) * contact
        return torch.sum(contact_vel_xy, dim=1)
    
    def _reward_keep_feet(self):
        # Penalize contact with no velocity
        pos_errors = torch.abs(self.dof_pos[:, self.hip_knee_dof_indices] - self.default_dof_pos[:, self.hip_knee_dof_indices])
        total_error = torch.sum(pos_errors, dim=1)
        return torch.exp(-total_error / 0.5)
    
    def _reward_both_feet_contact(self):
        """
        保持双脚持续触地的奖励函数
        设计要点：
        1. 忽略步态相位，直接检测物理接触
        2. 必须所有足端同时触地才给予奖励
        3. 加入接触力强度奖励，防止"轻触即走"
        """
        # ==================== 基础接触检测 ==================== 
        # 检测所有足端垂直接触力是否超过阈值（更严格的阈值）
        contact_threshold = 5.0  # 单位：牛顿，比原函数1N更严格防止误触发
        contact_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        for foot_idx in self.feet_indices:
            vertical_force = self.contact_forces[:, foot_idx, 2]
            contact_mask &= (vertical_force > contact_threshold)

        z_forces = self.contact_forces[:, self.feet_indices, 2]         
        # print("contact_forces shape:", self.contact_forces.shape)          # 应为 [N, B, 3]
        # print("z_forces shape:", z_forces.shape)                          # 应为 [N, F]

        
        # ==================== 接触稳定性增强 ====================
        # 计算接触力平方和（鼓励稳定接触）
        contact_force_sum = torch.sum(
            torch.square(z_forces), 
            dim=1  # 修正：只在足端维度求和
        )
        # print("contact_force_sum shape:", contact_force_sum.shape)        # 应为 [N]        
        # ==================== 时间连续性奖励 ====================
        # 记录接触持续时间（需在reset中初始化self.contact_duration）
        self.contact_duration = torch.where(
            contact_mask,
            self.contact_duration + self.dt,
            torch.zeros_like(self.contact_duration)
        )
        duration_bonus = torch.clamp(self.contact_duration / 2.0, 0.0, 1.0)  # 持续2秒达到最大奖励

        # ==================== 组合奖励 ====================
        base_reward = contact_mask.float()  # 基础接触奖励（0或1）
        force_bonus = torch.exp(-contact_force_sum / 500.0)  # 接触力强度奖励
        total_reward = (base_reward + 0.3 * force_bonus + 0.2 * duration_bonus) 
        
        return total_reward * 1  # 从配置读取权重
    
    # def _reward_both_feet_contact(self):

    #     contact_threshold = 5.0 #N
    #     contact_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
    #     for foot_idx in self.feet_indices:
    #         vertical_force = self.contact_forces[:, foot_idx, 2]
    #         contact_mask &= (vertical_force > contact_threshold)

    #     z_forces = self.contact_forces[:, self.feet_indices, 2]         
    #     contact_force_sum = torch.sum(
    #         torch.square(z_forces), 
    #         dim=1
    #     )
    #     self.contact_duration = torch.where(
    #         contact_mask,
    #         self.contact_duration + self.dt,
    #         torch.zeros_like(self.contact_duration)
    #     )
    #     duration_bonus = torch.clamp(self.contact_duration / 2.0, 0.0, 1.0)
    #     base_reward = contact_mask.float() 
    #     force_bonus = torch.exp(-contact_force_sum / 500.0)
    #     total_reward = (base_reward + 0.3 * force_bonus + 0.2 * duration_bonus) 
        
    #     return total_reward * 1
    # def _reward_both_feet_on_ground(self):
    #     res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    #     # Check if each foot is in stance phase (e.g., phase < 0.55)
    #     is_stance = self.leg_phase < 0.55  # shape: [num_envs, feet_num]
        
    #     # Check if each foot has ground contact (z force > 1)
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1  # shape: [num_envs, feet_num]

    #     # Correct contact means the foot is in stance AND has contact
    #     correct_contact = contact & is_stance

    #     # Count how many feet are correctly in contact
    #     num_correct_contacts = correct_contact.sum(dim=1)

    #     # Reward if BOTH feet are in correct contact
    #     res = (num_correct_contacts == self.feet_num).float()
        
    #     return res
    def _reward_stable_posture(self):
        """ 
        关节稳定姿态奖励函数 + 任务完成时间效率奖励
        任务完成条件：所有目标关节角度持续在阈值范围内
        """
        # ========== 1. 定义关节参数 ==========
        target_joints = {
            "hip": self.hip_dof_indices,    # 髋关节索引
            "knee": self.hip_knee_dof_indices,  # 膝关节索引
            "ankle": self.ankle_dof_indices # 踝关节索引
        }
        joint_indices = torch.cat([
        self.hip_dof_indices, 
        self.hip_knee_dof_indices, 
        self.ankle_dof_indices], dim=0)


        angle_threshold = 0.1  # 角度允许的阈值（±0.1弧度）
        stable_steps_required = 4  # 需连续稳定的步数
        
        # ========== 2. 检测关节是否在目标范围内 ==========
        # 计算各关节角度绝对误差
        joint_errors = torch.abs(self.dof_pos[:, joint_indices])
        
        # 判断每个关节是否在阈值内 [num_envs, num_joints]
        within_threshold = joint_errors < angle_threshold
        
        # 所有目标关节均需满足条件 [num_envs]
        all_joints_stable = torch.all(within_threshold, dim=1)
        
        # 更新连续稳定步数计数器
        self.stable_steps_counter = torch.where(
            all_joints_stable,
            self.stable_steps_counter + 1,
            torch.zeros_like(self.stable_steps_counter)
        )
        
        # 任务完成标志：连续稳定步数达标
        task_complete = self.stable_steps_counter >= stable_steps_required
        
        # ========== 3. 计算关节姿态奖励 ==========
        angle_errors = torch.sum(joint_errors, dim=1)
        posture_reward = torch.exp(-angle_errors / 0.2)  # 温度系数=0.2
        
        # ========== 4. 时间效率奖励 ==========
        # 累计未完成任务的时间（仅对未完成环境计时）
        self.elapsed_time += self.dt * (~task_complete).float()
        
        # 任务完成时的效率奖励（完成越早奖励越高）
        time_efficiency = torch.where(
            task_complete,
            1.0 / (self.elapsed_time + 1e-5),  # 防止除零
            torch.zeros_like(self.elapsed_time)
        )
        
        # ========== 5. 组合奖励 ==========
        total_reward = (
            0.6* posture_reward +
            0.4* time_efficiency
        )
        
        # 重置已完成任务的计时器和计数器
        self.elapsed_time[task_complete] = 0.0
        self.stable_steps_counter[task_complete] = 0
        
        return total_reward