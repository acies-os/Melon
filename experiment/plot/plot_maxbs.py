import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os


data_dir = 'data/evaluation/maxbs'
os.makedirs(data_dir, exist_ok=True)


def _plot(data, file):
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 5))

    confs = {
        "linewidth": 5,
        "color": ['grey', 'blue', 'black', 'red'],
        "markersize": 6
    }
    markers = ['>', 'v', 's', 'o']
    ax = data.plot(ax=ax, x=None, y=data.columns[1:], kind="line", **confs)

    for i, line in enumerate(ax.get_lines()):
        line.set_marker(markers[i])
        line.set_markersize(15)

    plt.legend(prop={'size': 20})
    # plt.xscale('log')
    ax.set_xticks(np.arange(len(data)))
    # ax.invert_xaxis()
    # ax.xaxis.set_major_locator(mtick.FixedLocator(data['throughput']))
    # ax.xaxis.set_major_locator(mtick.FixedLocator(range(int(data['throughput'].max()))))
    ax.set_xticklabels(data['throughput'])
    ax.set_xlabel('Throughput (fps)', fontsize=32)
    plt.xticks(fontsize=32)
    ax.set_ylabel('Maximal batch size', fontsize=32)

    # x_labels = [('BS=' + str(bs)) for bs in data['batch'].tolist()]
    # ax.set_xticklabels(x_labels, rotation=0, fontsize=20)

    plt.yticks(range(40, 201, 40), fontsize=32)

    plt.savefig(file, bbox_inches='tight', pad_inches=0)


def dump_log_file(model):
    if model == "MobilenetV2":
        batches = (112, 176, 208)
    else:
        batches = (96, 144, 176, 192)

    baseline_fn = f'output/mnn/{model}/{model}.32.mnn.out'
    latency = []
    with open(baseline_fn) as f:
        for line in f:
            if line.startswith('train, 187') and ', cost time:' in line:
                latency.append(float(line.strip().split(':')[1].split()[0].strip()))
    throughput = [32 * 1000 / np.mean(latency)]
    maxbs = [32]

    for batch in batches:
        melon_fn = f'output/ours/{model}/{model}.{batch}.5500.ours.out'
        latency = []
        with open(melon_fn) as f:
            for line in f:
                if line.startswith('train, 187') and ', cost time:' in line:
                    latency.append(float(line.strip().split(':')[1].split()[0].strip()))
        if len(latency):
            throughput.append(batch * 1000 / np.mean(latency))
            maxbs.append(batch)
    with open(os.path.join(data_dir, f'{model}.csv'), 'w') as f:
        f.write('throughput,MNN,Melon\n')
        for th, bs in zip(throughput, maxbs):
            f.write(f'{th:.2f},32,{bs}\n')


for model in ['Squeezenet', 'MobilenetV2']:
    dump_log_file(model)
    df = pd.read_csv(os.path.join(data_dir, f'{model}.csv'))
    df['throuthput'] = df['throughput'].map(str)
    # df = df[['batch', 'ideal', 'vdnn', 'oracle-swap', 'sublinear', 'capuchin', 'ours']]
    print(model, df, sep='\n')
    _plot(df, os.path.join('data/evaluation/maxbs',  f'{model}.pdf'))
