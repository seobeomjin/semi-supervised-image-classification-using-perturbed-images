
"train cvae"  
python train_cvae.py --config cifar100_StrongAug.json --gpu-id 2 --dataset cifar100_for_cvae_train
nohup python train_cvae.py --config cifar100_StrongAug.json --gpu-id 2 --dataset cifar100_for_cvae_train > experiments/cifar100_StrongAug/nodup.out &

python train_cvae.py --config cifar100_StrongAug.json --gpu-id 2 --dataset cifar100_for_cvae_train --resume experiments/cifar100_StrongAug/checkpoints/checkpoint_499.pth




"train classifier" 
# Train FixMatch + cvae-attack, cifar10@4000.5
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/perturbed_cifar10@4000.5 --config-robust cifar10_wideresnet_cvae_attack.json
# Train FixMatch + cvae-attack, cifar10@4000.5 with nohup  
nohup python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/perturbed_cifar10@4000.5 --config-robust cifar10_wideresnet_cvae_attack.json > results/perturbed_cifar10@4000.5.out & 
# Train FIxMatch + cvae-attack, resume, cifar10@4000.5
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/perturbed_cifar10@4000.5 --config-robust cifar10_wideresnet_cvae_attack.json --resume results/try1_perturbed_cifar10@4000.5/model_best.pth.tar


# Train FixMatch + cvae aug , cifar10@4000.5 
    # >>> 1) I made inputs_u_ptb via pseudo_label_ got from model(inputs_u_w) 
    # >>> 2) and then train model with inputs(=x, u_w, u_ptb)
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --total-step 262144 --lr 0.03 --expand-labels --gpu-id 3 --seed 5 --out results/cvae_aug_cifar10@4000.5_test --config-robust configs/cifar10_wideresnet_cvae_aug.json
# Train FixMatch + cvae-aug, cifar10@4000.5 with nohup 
nohup python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --total-step 262144 --lr 0.03 --expand-labels --gpu-id 1 --seed 5 --out results/cvae_aug_cifar10@4000.5 --config-robust cifar10_wideresnet_cvae_aug.json > results/cvae_aug_cifar10@4000.5.out &


# Train FixMatch + cvae smoothing 084 , cifar10@4000.5 
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --total-step 262144 --lr 0.03 --expand-labels --gpu-id 2 --seed 5 --out results/cvae_smoothing_084_cifar10@4000.5 --config-robust cifar10_wideresnet_cvae_smoothing_084.json
# Train FixMatch + cvae smoothing 084 with nohup 
nohup python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --total-step 262144 --lr 0.03 --expand-labels --gpu-id 2 --seed 5 --out results/cvae_smoothing_084_cifar10@4000.5 --config-robust cifar10_wideresnet_cvae_smoothing_084.json > results/cvae_smoothing_084_cifar10@4000.5.out &


# Train FixMatch + cvae smoothing 122 , cifar10@4000.5 
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --total-step 262144 --lr 0.03 --expand-labels --gpu-id 3 --seed 5 --out results/cvae_smoothing_122_cifar10@4000.5 --config-robust cifar10_wideresnet_cvae_smoothing_122.json
# Train FixMatch + cvae smoothing 122 with nohup 
nohup python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --total-step 262144 --lr 0.03 --expand-labels --gpu-id 3 --seed 5 --out results/cvae_smoothing_122_cifar10@4000.5 --config-robust cifar10_wideresnet_cvae_smoothing_122.json > results/cvae_smoothing_122_cifar10@4000.5.out &


# Train FixMatch + cvae aug , cifar100@400.5
python train.py --gpu-id 2 --dataset cifar100 --num-labeled 400 --arch wideresnet --batch-size 16 --total-step 262144 --lr 0.03 --wdecay 0.001 --expand-labels --seed 5 --out results/cifar100@400-2 --config-robust cifar100_wideresnet_cvae_aug.json
python train.py --gpu-id 3 --dataset cifar100 --num-labeled 400 --arch wideresnet --batch-size 16 --total-step 262144 --lr 0.03 --wdecay 0.001 --expand-labels --seed 5 --out results/cifar100@400 --config-robust cifar100_wideresnet_cvae_aug.json --resume results/cifar100@400/checkpoint.pth.tar

# Train FixMatch + cvae aug , cifar100@400.5 with nohup
nohup python train.py --gpu-id 2 --dataset cifar100 --num-labeled 400 --arch wideresnet --batch-size 16 --total-step 262144 --lr 0.03 --wdecay 0.001 --expand-labels --seed 5 --out results/cifar100@400 --config-robust cifar100_wideresnet_cvae_aug.json --resume results/cifar100@400/checkpoint.pth.tar > results/cifar100@400.5-resume.out &

###python train2.py --gpu-id 2 --dataset cifar100 --num-labeled 400 --arch wideresnet --batch-size 64 --total-step 262144 --lr 0.03 --wdecay 0.001 --expand-labels --seed 5 --out results/cifar100@400 --config-robust cifar100_wideresnet_cvae_aug.json


