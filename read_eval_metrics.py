import joblib

file_path = "/nethome/mkhanna38/disk/PointNav-VO/train_log/seed_100-vo-320_noise_0-train-rgb_d-dd_none_10-m_cen_1-act_1-model_vo_cnn-resnet18-geo__inv_w_1-l_mult_fix_1-1.0_1.0_1.0-dpout_0.2-e_150-b_128-lr_0.00025-w_de_0.0-20211101_022917802735/checkpoints/eval_20211103_231014960483/infos/eval_forward_regression_info.p"
# file_path = '/nethome/mkhanna38/disk/PointNav-VO/train_log/seed_100-vo-320_noise_0-train-rgb_d-dd_none_10-m_cen_1-act_2_3-model_vo_cnn-resnet18-geo_inv_joint_inv_w_1-l_mult_fix_1-1.0_1.0_1.0-dpout_0.2-e_150-b_128-lr_0.00025-w_de_0.0-20211102_130227063106/checkpoints/eval_20211104_055024850629/infos/eval_left_cur_rel_to_prev_regression_info.p'

f = joblib.load(file_path)
print(file_path)
print("abs_diff_dz", f["abs_diff_dz"])
print("relative_diff_dz", f["relative_diff_dz"])
print("---------")
print("abs_diff_dx", f["abs_diff_dx"])
print("relative_diff_dx", f["relative_diff_dx"])
print("---------")
print("abs_diff_dyaw", f["abs_diff_dyaw"])
print("relative_diff_dyaw", f["relative_diff_dyaw"])
