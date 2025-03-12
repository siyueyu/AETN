import os

filepath = '/home/yusiyue/project/MCTformer/MCTformer_results/MCTformer_ecai/coco/fused-patchrefine-npy'

for root, dirs, files in os.walk(filepath):
    for file in files:
        f = open('/home/yusiyue/project/MCTformer/coco/train_eval_id.txt', 'a')
        f.write(file[:-4] + '\n')
        f.close()