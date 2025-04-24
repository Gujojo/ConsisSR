# 参数列表
params=(5.0)

# 固定参数
A="/home/gujh/sdsr/RealSR/test_HR"
E=1

# 循环遍历参数列表
for param in "${params[@]}"
do
    B="configs/model/cldm.yaml"
    C="/home/gujh/sdsr/ConsisSR/logs/uniform_hybrid_lqinput_cpip1_ImageNet_filter_c_/step=29999.ckpt"
    D="/home/gujh/sdsr/RealSR/ConsisSR_wocfg${param}_tmp"
    CUDA_VISIBLE_DEVICES='1' python inference.py --sr_scale 1 --input $A --config $B --ckpt $C --output $D --cfg_scale $param --repeat_times $E
done

# # 固定参数
# A="/my_project_path/StableSR_testsets/DIV2K_V2_val/lq"
# E=1

# # 循环遍历参数列表
# for param in "${params[@]}"
# do
#     B="configs/model/cldm_ori_clip_distill.yaml"

#     C="/my_project_path/DiffBIR-main/logs/uniform_hybrid_lqinput_cpip1_toip_ImageNet_c/lightning_logs/version_0/checkpoints/step=49999.ckpt"
#     D="/my_project_path/results/DIV2K_cfg/uh_lq_cpip1_ImageNet_c_200k_wocfg${param}"
#     CUDA_VISIBLE_DEVICES='1' python inference_diff.py --input $A --config $B --ckpt $C --output $D --cfg_scale $param --repeat_times $E
# done

# python iqa.py