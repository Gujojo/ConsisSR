# 参数列表
params=(4.0)

# # 固定参数
# A="/my_project_path/StableSR_testsets/DrealSRVal_crop128/test_LR"
# E=1

# # 循环遍历参数列表
# for param in "${params[@]}"
# do
#     B="configs/model/cldm_ori_clip_distill.yaml"
#     C="/my_project_path/DiffBIR-main/logs/uniform_hybrid_lqinput_cpip1_toip_ImageNet/lightning_logs/version_0/checkpoints/step=149999.ckpt"
    
#     D="/my_project_path/results/DrealSR_lre/uh_lq_cpip1_ImageNet_90_s10_150k_wocfg${param}"
#     CUDA_VISIBLE_DEVICES='0' python inference_lre.py --input $A --config $B --ckpt $C --output $D --cfg_scale $param --repeat_times $E --steps 10 --vq /my_project_path/results/DrealSR/History/VAE_vqpcg_512_retrain

#     D="/my_project_path/results/DrealSR_lre/uh_lq_cpip1_ImageNet_90_s20_150k_wocfg${param}"
#     CUDA_VISIBLE_DEVICES='0' python inference_lre.py --input $A --config $B --ckpt $C --output $D --cfg_scale $param --repeat_times $E --steps 20 --vq /my_project_path/results/DrealSR/History/VAE_vqpcg_512_retrain
# done


# 固定参数
A="/my_project_path/StableSR_testsets/RealSRVal_crop128/test_LR"
E=1

# 循环遍历参数列表
for param in "${params[@]}"
do
    B="configs/model/cldm_ori_clip_distill.yaml"
    C="/my_project_path/DiffBIR-main/logs/uniform_hybrid_lqinput_cpip1_toip_ImageNet/lightning_logs/version_0/checkpoints/step=149999.ckpt"

    D="/my_project_path/results/RealSR_lre/uh_lq_cpip1_ImageNet_90_s10_150k_wocfg${param}"
    CUDA_VISIBLE_DEVICES='1' python inference_lre.py --input $A --config $B --ckpt $C --output $D --cfg_scale $param --repeat_times $E  --steps 10 --vq /my_project_path/results/RealSR/VAE_vqpcg_512_retrain_90k

    D="/my_project_path/results/RealSR_lre/uh_lq_cpip1_ImageNet_90_s20_150k_wocfg${param}"
    CUDA_VISIBLE_DEVICES='1' python inference_lre.py --input $A --config $B --ckpt $C --output $D --cfg_scale $param --repeat_times $E  --steps 20 --vq /my_project_path/results/RealSR/VAE_vqpcg_512_retrain_90k
done
