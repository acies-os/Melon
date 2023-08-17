import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

data_dir = 'data/evaluation/throughput'
os.makedirs(data_dir, exist_ok=True)

def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        if y_value > 0: continue

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        label = 'X'

        # Create annotation
        ax.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(0, space),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',  # Horizontally center label
            va=va, fontsize=18)  # Vertically align label differently for
        # positive and negative values.


def _plot(data, file):
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 5))

    confs = {
        "edgecolor": 'black',
        "linewidth": 1.5,
        "width": 0.75,
        # 'hatch': ['/', '\\', '|', '-', '+']
        # "color": ['white', 'grey', 'yellow', 'green', 'blue', 'black']
    }
    hatches = ['/', '\\', '|', '-', 'x']
    ax = data.plot(ax=ax, x="batch", y=data.columns[1:], kind="bar", **confs)
    hatches = ''.join(h * len(df) for h in hatches)
    # return
    bars = ax.patches
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    height = data['ours'].max()
    if height - int(height) <= 0.5:
        height = int(height) + 0.5
    else:
        height = int(height) + 1

    # if height == 1:
    # 	plt.yticks(np.arange(0, 1.6, 0.5), fontsize=30)
    # elif height == 2:
    # 	plt.yticks(np.arange(0, 2, 0.5), fontsize=30)
    # elif height == 3:
    # 	plt.yticks(np.arange(0, 3.1, 1), fontsize=30)
    # elif height == 4:
    # 	plt.yticks(np.arange(0, 5, 1), fontsize=30)
    # plt.yticks(np.arange(0, height, height/4.0), fontsize=30)
    if all(map(lambda x: x in file, ['Resnet', 'RedmiNote8', 'nobn'])):
        plt.yticks(np.arange(0, 0.7, 0.2), fontsize=30)
    else:
        plt.yticks(np.arange(0, height + 0.1, height / 5), fontsize=30)

    # X = np.arange(4)
    # plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    # plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
    # plt.bar(X + 0.50, data[2], color = 'r', width = 0.25)

    # plt.legend()
    # plt.xscale('log')
    # ax.set_xlabel('response delay (ms)', fontsize=15)
    # plt.xticks(fontsize=13)
    ax.set_xlabel('')
    ax.set_ylabel('Throughput (fps)', fontsize=30)

    add_value_labels(ax)

    x_labels = [('BS=' + str(bs)) for bs in data['batch'].tolist()]
    ax.set_xticklabels(x_labels, rotation=0, fontsize=30)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    # if 'Resnet' in file:
    # 	ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
    # else:
    # 	ax.yaxis.set_major_locator(mtick.MultipleLocator(1))

    # ax.get_legend().remove()

    plt.savefig(file, bbox_inches='tight', pad_inches=0)


def dump_log_file(model):
    if model == 'Resnet50':
        batches = (32, 64)
    else:
        batches = (64, 96, 128)

    mnn_log = f'output/mnn/{model}/{model}.8.mnn.out'
    latency = []
    with open(mnn_log) as f:
        for line in f:
            if line.startswith('train, 187') and ', cost time:' in line:
                latency.append(float(line.strip().split(':')[1].split()[0].strip()))
    mnn_throughput = 8 * 1000 / np.mean(latency)

    throughput = []
    for batch in batches:
        melon_fn = f'output/ours/{model}/{model}.{batch}.5500.ours.out'
        latency = []
        with open(melon_fn) as f:
            for line in f:
                if line.startswith('train, 187') and ', cost time:' in line:
                    latency.append(float(line.strip().split(':')[1].split()[0].strip()))
        if len(latency):
            throughput.append(batch * 1000 / np.mean(latency))
        else:
            throughput.append(0)

    with open(os.path.join(data_dir, f'{model}.csv'), 'w') as f:
        f.write('batch,ours,ideal\n')
        for batch, th in zip(batches, throughput):
            f.write(f'{batch},{th},{mnn_throughput}\n')


for model in ["MobilenetV2", "Squeezenet", "MobilenetV1", "Resnet50"]:
    dump_log_file(model)
    df = pd.read_csv(os.path.join(data_dir, f'{model}.csv'))
    # df = df[['batch', 'ours', 'ideal']]
    # for sys in df.columns[1:]:
    #     df[sys] = df.apply(lambda row: 1000/row[sys]*row['batch'], axis=1)
    # print (df)
    # device = f.split('-')[1]
    # print('vdnn', df['ours'] / df['vdnn'])
    # print('sublinear', df['ours'] / df['sublinear'])
    # print('capuchin', df['ours'] / df['capuchin'])
    # speedup.extend([a for a in df['ours'] / df['vdnn'] if str(a) != 'nan'])
    # speedup.extend([a for a in df['ours'] / df['sublinear'] if str(a) != 'nan'])
    # speedup.extend([a for a in df['ours'] / df['capuchin'] if str(a) != 'nan'])
    # plot_data_dir = os.path.join(data_dir, device)
    # if not os.path.exists(plot_data_dir): os.mkdir(plot_data_dir)
    _plot(df, os.path.join(data_dir, model+'.pdf'))