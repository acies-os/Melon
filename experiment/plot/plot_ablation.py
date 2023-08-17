import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

data_dir = 'data/evaluation/ablation'
os.makedirs(data_dir, exist_ok=True)

bar_width = 0.3
lw = 3


def dump_log_file(model):
    # ours
    if model == 'MobilenetV2':
        batches = [112, 176]
    else:
        batches = [96, 144]

    throughput_ours = []
    for batch in batches:
        melon_fn = f'output/ours/{model}/{model}.{batch}.5500.ours.out'
        latency = []
        with open(melon_fn) as f:
            for line in f:
                if line.startswith('train, 187') and ', cost time:' in line:
                    latency.append(float(line.strip().split(':')[1].split()[0].strip()))
        throughput_ours.append(batch * 1000 / np.mean(latency))

    # recompute
    if model == 'MobilenetV2':
        batches = [80, 96]
    else:
        batches = [48, 80]

    throughput_recomp = []
    for batch in batches:
        melon_fn = f'output/ours/{model}/{model}.{batch}.5500.ours.recompute.out'
        latency = []
        with open(melon_fn) as f:
            for line in f:
                if line.startswith('train, 187') and ', cost time:' in line:
                    latency.append(float(line.strip().split(':')[1].split()[0].strip()))
        throughput_recomp.append(batch * 1000 / np.mean(latency))

    # pool
    if model == 'MobilenetV2':
        batches = [40, 40]
    else:
        batches = [48, 48]

    throughput_pool = []
    for batch in batches:
        melon_fn = f'output/ours/{model}/{model}.{batch}.5500.ours.pool.out'
        latency = []
        with open(melon_fn) as f:
            for line in f:
                if line.startswith('train, 187') and ', cost time:' in line:
                    latency.append(float(line.strip().split(':')[1].split()[0].strip()))
        throughput_pool.append(batch * 1000 / np.mean(latency))

    tp1, tp2 = min(throughput_ours[0], throughput_recomp[0], throughput_pool[0]), min(throughput_ours[1], throughput_recomp[1], throughput_pool[1])
    with open(os.path.join(data_dir, f'{model}.csv'), 'w') as f:
        f.write('throughput,MNN,pool,recompute,Melon\n')
        if model == 'MobilenetV2':
            f.write(f'{tp1:.2f},32,40,80,112\n')
            f.write(f'{tp2:.2f},32,40,80,176\n')
        else:
            f.write(f'{tp1:.2f},32,48,64,96\n')
            f.write(f'{tp2:.2f},32,48,80,144\n')


for model in ['MobilenetV2', 'Squeezenet']:
    plt.figure(figsize=[9, 7])
    hatches = '/\\|-x'
    dump_log_file(model)
    df = pd.read_csv(os.path.join(data_dir, f'{model}.csv'))
    for i, tag in enumerate(['MNN', 'pool', 'recompute', 'Melon']):
        plt.bar(x=np.arange(0, 4, 2) + bar_width * i, height=df[tag], hatch=hatches[i], width=bar_width, label=tag, linewidth=lw, edgecolor='k')
    # plt.bar(x=np.arange(0, 4, 2), height=df['MNN'], width=bar_width, label='MNN', linewidth=lw, edgecolor='k')
    # plt.bar(x=np.arange(0, 4, 2) + bar_width, height=df['pool'], width=bar_width, label='pool', linewidth=lw, edgecolor='k')
    # plt.bar(x=np.arange(0, 4, 2) + bar_width * 2, height=df['recompute'], width=bar_width, label='recomputation', linewidth=lw, edgecolor='k')
    # plt.bar(x=np.arange(0, 4, 2) + bar_width * 3, height=df['CAMEL'], width=bar_width, label='CAMEL', linewidth=lw, edgecolor='k')

    if model == 'MobilenetV2':
        plt.xticks(np.arange(0, 4, 2)+bar_width*1.5, df['throughput'].tolist(), fontsize=30)
    else:
        plt.xticks(np.arange(0, 4, 2)+bar_width*1.5, df['throughput'].tolist(), fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel('Throughput (fps)', fontsize=30)
    plt.ylabel('Maximun batch size', fontsize=30)
    plt.legend()
    plt.savefig(os.path.join(data_dir, model+'.pdf'), bbox_inches='tight', pad_inches=0)

