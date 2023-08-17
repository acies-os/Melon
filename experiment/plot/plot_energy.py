import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

data_dir = 'data/evaluation/energy'
os.makedirs(data_dir, exist_ok=True)


def get_latency(filename):
    latency = []
    with open(filename) as f:
        for line in f:
            if line.startswith('train, 187') and ', cost time:' in line:
                latency.append(float(line.strip().split(':')[1].split()[0].strip()))
    return np.mean(latency)



def dump_log_file(model):
    usb_current, usb_voltage, batt_current, batt_voltage = [], [], [], []
    with open('energy.out') as f:
        pre_pid = None
        for line in f:
            pid, u_c, u_v, b_c, b_v = line.strip('\n').split('\t')
            if pre_pid != pid:
                pre_pid = pid
                # print(f'[{pre_pid}],[{pid}]')
                usb_current.append([])
                usb_voltage.append([])
                batt_current.append([])
                batt_voltage.append([])
            usb_current[-1].append(int(u_c))
            usb_voltage[-1].append(int(u_v))
            batt_current[-1].append(int(b_c))
            batt_voltage[-1].append(int(b_v))

    powers = []
    for i in range(len(usb_current)):
        power = (np.mean(usb_current[i]) * np.mean(usb_voltage[i]) + np.mean(batt_current[i]) * np.mean(batt_voltage[i])) / 1e12
        powers.append(power)
    tags = [
        ("MobilenetV2", 'mnn', 32),
        ("Squeezenet", 'mnn', 32),

        ("MobilenetV2", 'ours', 64),
        ("MobilenetV2", 'ours', 96),
        ('Squeezenet', 'ours', 64),
        ('Squeezenet', 'ours', 96),
    ]

    with open(os.path.join(data_dir, f'{model}.csv'), 'w') as f:
        f.write('batch,ideal,Melon\n')
        if model == 'MobilenetV2':
            f.write(f'64,{powers[0] * get_latency(f"output/mnn/{model}/{model}.32.mnn.out") * 2},'
                    f'{powers[2] * get_latency(f"output/ours/{model}/{model}.64.5500.ours.out")}\n')
            f.write(f'96,{powers[0] * get_latency(f"output/mnn/{model}/{model}.32.mnn.out") * 3},'
                    f'{powers[3] * get_latency(f"output/ours/{model}/{model}.96.5500.ours.out")}\n')
        else:
            f.write(f'64,{powers[1] * get_latency(f"output/mnn/{model}/{model}.32.mnn.out") * 2},'
                    f'{powers[4] * get_latency(f"output/ours/{model}/{model}.64.5500.ours.out")}\n')
            f.write(f'96,{powers[1] * get_latency(f"output/mnn/{model}/{model}.32.mnn.out") * 3},'
                    f'{powers[5] * get_latency(f"output/ours/{model}/{model}.96.5500.ours.out")}\n')


for model in ['MobilenetV2', 'Squeezenet']:
    dump_log_file(model)
    fig, ax = plt.subplots(figsize=(8, 6))
    df = pd.read_csv(os.path.join(data_dir, f'{model}.csv'))
    keys = ['ideal', 'sublinear', 'vdnn', 'capuchin', 'ours']
    ideal = list(map(float, df['ideal'] / df['ideal']))
    # sublinear = list(map(float, df['sublinear'] / df['ideal']))
    # vdnn = list(map(float, df['vdnn'] / df['ideal']))
    # capuchin = list(map(float, df['capuchin'] / df['ideal']))
    ours = list(map(float, df['Melon'] / df['ideal']))
    data = [ours, ideal]
    labels = ['Melon', 'ideal']
    # print(ideal, sublinear, vdnn, capuchin, ours, sep='\n')
    # print()
    # print(np.array(ours) - np.array(sublinear))
    # print(np.array(ours) - np.array(vdnn))
    # print(np.array(ours) - np.array(capuchin))
    # print()
    bar_width = 0.3
    lw = 3
    hatches = '/\\|-x'
    for i in range(len(data)):
        plt.bar(x=np.arange(0, 4, 2) + i*bar_width, height=data[i], width=bar_width, hatch=hatches[i], label=labels[i], linewidth=lw, edgecolor='k')
    # plt.bar(x=np.arange(0, 4, 2), height=vdnn, width=bar_width, label='ideal', linewidth=lw, edgecolor='k')
    # plt.bar(x=np.arange(0, 4, 2) + bar_width, height=sublinear, width=bar_width, label='vdnn', linewidth=lw, edgecolor='k')
    # plt.bar(x=np.arange(0, 4, 2) + bar_width * 2, height=capuchin, width=bar_width, label='sublinear', linewidth=lw, edgecolor='k')
    # plt.bar(x=np.arange(0, 4, 2) + bar_width * 3, height=ours, width=bar_width, label='capuchin', linewidth=lw, edgecolor='k')
    # plt.bar(x=np.arange(0, 4, 2) + bar_width * 4, height=ideal, width=bar_width, label='ours', linewidth=lw, edgecolor='k')
    plt.xticks(np.arange(0, 4, 2)+bar_width*2, ['BS=64', 'BS=96'], fontsize=30)
    plt.yticks(fontsize=30)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
    plt.text(2 + bar_width*0.85, 0.05, 'X', fontsize=18)
    plt.legend()
    plt.ylabel('Normalized Energy', fontsize=30)
    # plt.xlabel('batchsize', fontsize=20)
    plt.savefig(os.path.join(data_dir, f'{model}.pdf'), bbox_inches='tight', pad_inches=0)