from queue import Queue
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--latency", type=float, required=False, default=0, help="The memcpy latency")
args = parser.parse_args()

root = '.'
data_dir = 'data/evaluation/adaptive'
os.makedirs(data_dir, exist_ok=True)
bar_width = 0.3
lw = 3


class Profiler:
    alignment = 64

    def __init__(self, model, batch):
        self.model = model
        self.batch = batch
        self.tensor_size = {}
        self.io_info = []
        self.resize_info = []
        self.redundent_parent = {}  # 存在不断alloc free alloc free的情况，这种应该是同一个tensor，去重
        self.cost_info = {}
        self.tensor_from_opid = {}
        self.fp_thres = -1
        self.num_layers = 0

        if self.model == 'Googlenet':
            self.fp_thres = 1849
        elif self.model == 'MobilenetV2' or self.model == 'MobilenetV2_CL':
            self.fp_thres = 1750
            self.num_layers = 30
        elif self.model == 'MobilenetV1':
            self.num_layers = 31
            self.fp_thres = 917
        elif self.model == 'Squeezenet' or self.model == 'Squeezenet_CL':
            self.fp_thres = 866
            self.num_layers = 30
        elif self.model == 'Resnet50':
            self.fp_thres = 1835
            self.num_layers = 50
        elif self.model == 'MobilenetV1NoBN':
            self.fp_thres = 121
            self.num_layers = 31
        elif self.model == 'MobilenetV2NoBN':
            self.fp_thres = 227
            self.num_layers = 30
        elif self.model == 'SqueezenetNoBN':
            self.fp_thres = 120
            self.num_layers = 30
        elif self.model == 'Resnet50NoBN':
            self.fp_thres = 298
            self.num_layers = 50
        # self.profile()
        # self.resize()

    def profile(self, fpath: str = ''):
        if not fpath:
            fpath = os.path.join(root, 'profile', self.model, f'{self.model}.{self.batch}.profile.out')
        if not os.path.exists(fpath):
            return

        def add_info(ln, tag):
            nonlocal self
            ln = ln.strip().split(':')[-1].strip().replace('[', '').replace(']', '').strip().split(',')
            tmp = set()
            for item in ln:
                if len(item):
                    item = item.strip().replace('(', '').replace(')', '').split()
                    tid, tsize = int(item[0]), int(item[1])
                    self.tensor_size[tid] = tsize
                    tmp.add(tid)
            self.io_info[-1][tag] = list(tmp)

        profile_flag = False
        with open(fpath) as f:
            for line in f:
                if line.strip().endswith('start read-map'):
                    profile_flag = True
                elif line.strip().endswith('finish read-map & start replace'):
                    profile_flag = False
                if not profile_flag:
                    continue

                if line.strip().startswith('current Op'):
                    op = line.strip().split()[-1]
                    opid = len(self.io_info)
                    self.io_info.append({'op': op, 'id': opid})
                elif line.startswith('\t') and line.strip().startswith('outputs'):
                    add_info(line, 'outputs')
                    for t in self.io_info[-1]['outputs']:
                        self.tensor_from_opid[t] = len(self.io_info) - 1
                elif line.startswith('\t') and line.strip().startswith('release'):
                    add_info(line, 'release')
                elif line.startswith('\t') and line.strip().startswith('inputs'):
                    add_info(line, 'inputs')
                elif line.startswith('\t') and line.strip().startswith('temporary'):
                    add_info(line, 'temporary')
                    for t in self.io_info[-1]['temporary']:
                        assert t in self.io_info[-1]['outputs']

    def resize(self, fpath: str = ''):
        if not fpath:
            fpath = os.path.join(root, 'resize', self.model, f'{self.model}.{self.batch}.resize.out')
        if not os.path.exists(fpath):
            return

        resize_flag = False
        opid = None
        resize_tid = 0
        compute_flag = False
        freed = None
        with open(fpath) as f:
            for line in f:
                if 'start read-map' in line:
                    compute_flag = True
                if 'finish read-map' in line:
                    compute_flag = False
                if not compute_flag:
                    continue

                if line.strip().startswith('current Op is'):
                    opid = int(line.strip().split()[-1].split('th')[0])
                    self.tensor_size[opid] = 0
                    # print(opid)
                if line.strip().startswith('finish allocate memory for cmd'):
                    resize_flag = True
                    # print(opid, len(self.resize_info))
                    assert opid == len(self.resize_info)
                    self.resize_info.append([])
                    freed = set()  # freed tensors in current resize process
                if line.strip().startswith('try get'):
                    if resize_flag:
                        size = (int(line.strip().split()[2]) + Profiler.alignment - 1) // Profiler.alignment * Profiler.alignment
                        rtid = f'{opid}:{resize_tid}'
                        self.tensor_size[rtid] = size
                        redundent = None
                        for t in freed:
                            if self.tensor_size[t] == size:
                                freed.remove(t)
                                redundent = t
                                break
                        if redundent:
                            for i in range(len(self.resize_info[-1]) - 1, -1, -1):
                                if self.resize_info[-1][i][1] == redundent:
                                    self.resize_info[-1].pop(i)
                            self.redundent_parent[redundent] = rtid
                        self.resize_info[-1].append(('alloc', rtid))
                        resize_tid += 1
                    else:
                        self.tensor_size[opid] = (int(line.strip().split()[2]) + Profiler.alignment - 1) // Profiler.alignment * Profiler.alignment
                if line.strip().startswith('try return') and resize_flag:
                    size = (int(line.strip().split()[2]) + Profiler.alignment - 1) // Profiler.alignment * Profiler.alignment
                    talloc, tfree = [], []
                    for a, tid in self.resize_info[-1]:
                        if a == 'alloc' and self.tensor_size[tid] == size:
                            talloc.append(tid)
                        if a == 'free' and self.tensor_size[tid] == size:
                            tfree.append(tid)
                    # print(opid, self.resize_info[opid], [self.tensor_size[t] for a, t in self.resize_info[opid] if a == 'alloc'], size)
                    tid = [t for t in talloc if t not in tfree][-1]
                    self.resize_info[-1].append(('free', tid))
                    freed.add(tid)
                if line.strip().startswith('finish resize cmd'):
                    resize_flag = False
                    resize_tid = 0

    def cost(self, fpath: str = ''):
        if not fpath:
            fpath = os.path.join(root, 'cost', self.model, f'{self.model}.{self.batch}.cost.out')
        if not os.path.exists(fpath):
            return
        # print(fpath)
        cost_flag = False
        with open(fpath) as f:
            for line in f:
                # print(line)
                if line.strip().endswith('start read-map'):
                    cost_flag = True
                elif line.strip().endswith('finish read-map & start replace'):
                    cost_flag = False
                if not cost_flag:
                    continue

                if line.strip().startswith('current Op'):
                    op = line.strip().split()[-1]
                    opid = int(op.split('th')[0])
                    # print(opid)
                elif 'cost time' in line:
                    self.cost_info[opid] = float(line.strip().split()[-2])

    def load_infos(self):
        with open(os.path.join('data/profiler', self.model, f'{self.model}.io_info.json'), 'r') as f:
            self.io_info = json.load(f)
        for i in range(len(self.io_info)):
            for t in self.io_info[i]['outputs']:
                self.tensor_from_opid[t] = i
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.resize_info.json'), 'r') as f:
            self.resize_info = json.load(f)
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.tensor_size.json'), 'r') as f:
            self.tensor_size = json.load(f)
        for t in list(self.tensor_size.keys()):
            if ':' not in t:
                self.tensor_size[int(t)] = self.tensor_size.pop(t)
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.redundent_parent.json'), 'r') as f:
            self.redundent_parent = json.load(f)
        with open(f'data/profiler/{self.model}/{self.model}.{self.batch}.cost_info.json') as f:
            self.cost_info = json.load(f)
        for t in list(self.cost_info.keys()):
            self.cost_info[int(t)] = self.cost_info[t]

    def init_from_scratch(self):
        self.profile()
        self.resize()
        self.cost()
        os.makedirs(os.path.join('data/profiler', self.model), exist_ok=True)
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.tensor_size.txt'), 'w') as f:
            for t, tsz in self.tensor_size.items():
                f.write(f'{t}\t{tsz}\n')

    def dump_infos(self):
        with open(os.path.join('data/profiler', self.model, f'{self.model}.io_info.json'), 'w') as f:
            json.dump(self.io_info, f, indent=2)
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.resize_info.json'), 'w') as f:
            json.dump(self.resize_info, f, indent=2)
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.tensor_size.json'), 'w') as f:
            json.dump(self.tensor_size, f, indent=2)
        with open(os.path.join('data/profiler', self.model, f'{self.model}.{self.batch}.redundent_parent.json'), 'w') as f:
            json.dump(self.redundent_parent, f, indent=2)
        with open(f'data/profiler/{self.model}/{self.model}.{self.batch}.cost_info.json', 'w') as f:
            json.dump(self.cost_info, f, indent=2)


def adaptiveness(model, batch, previous_budget_gb, new_budget_gb):
    print(model, batch, previous_budget_gb, new_budget_gb, sep='\t')
    memcpy_speed = 22238208  # B/s
    if args.latency:
        memcpy_speed = 128 * 1024**2 * 4 * 100 / args.latency

    profiler = Profiler(model, batch)
    profiler.init_from_scratch()
    pre_plan_bgt, new_plan_bgt = None, None
    if previous_budget_gb == 6:
        pre_plan_bgt = 3800
    elif previous_budget_gb == 8:
        pre_plan_bgt = 5500

    if new_budget_gb ==  6:
        new_plan_bgt = 3800
    elif new_budget_gb == 8:
        new_plan_bgt = 5500

    if not pre_plan_bgt or not new_plan_bgt:
        print('fail')
        return None

    with open(f'heuristic/execution/{model}/{model}.{batch}.{pre_plan_bgt}.execution.txt') as f:
        pre_exe_seq = []
        for l in f:
            l = l.strip().split()
            pre_exe_seq.append([l[0], int(l[1])])
    with open(f'heuristic/execution/{model}/{model}.{batch}.{new_plan_bgt}.execution.txt') as f:
        new_exe_seq = []
        for l in f:
            l = l.strip().split()
            new_exe_seq.append([l[0], int(l[1])])
    with open(f'heuristic/allocation/{model}/{model}.{batch}.{pre_plan_bgt}.address.txt') as f:
        pre_address = {}
        for l in f:
            l = l.strip().split()
            pre_address[l[0]] = int(l[1])
    with open(f'heuristic/allocation/{model}/{model}.{batch}.{new_plan_bgt}.address.txt') as f:
        new_address = {}
        for l in f:
            l = l.strip().split()
            pre_address[l[0]] = int(l[1])

    adapt_results = {}
    for change_point in (0.25, 0.5, 0.75):
        cost_old_plan = 0
        allocated_tensor = set()
        for i in range(int(len(pre_exe_seq) * change_point)):
            a, op = pre_exe_seq[i]
            if 'compute' in a:
                cost_old_plan += profiler.cost_info[op]
                allocated_tensor.add(op)
            else:
                allocated_tensor.remove(op)
        reserve_tensor = set()
        for t in allocated_tensor:
            if pre_address[str(t)] + profiler.tensor_size[t] <= new_budget_gb * 1024 ** 3:
                reserve_tensor.add(t)
        if not reserve_tensor:
            print('oh my gooooooooooooood~')
            continue
        next_to_compute = max(reserve_tensor) + 1
        op_must_be_computed = set()
        cost_new_plan = 0
        for a, op in new_exe_seq:
            if a == 'compute' and op == next_to_compute:
                break
            if 'compute' in a:
                cost_new_plan += profiler.cost_info[op]
                op_must_be_computed.add(op)
            else:
                op_must_be_computed.remove(op)
        op_not_computed = op_must_be_computed - reserve_tensor
        op_computed = op_must_be_computed & reserve_tensor

        que = Queue()
        comp_src = set()
        for op in op_not_computed:
            que.put(op)
        while not que.empty():
            cur = que.get()
            if cur not in op_computed and cur not in comp_src:
                comp_src.add(cur)
                for ipt in profiler.io_info[cur]['inputs']:
                    que.put(ipt)

        memcpy_overhead = 3 * sum([profiler.tensor_size[t] for t in op_computed]) / memcpy_speed
        compute_overhead = 0
        for a, t in new_exe_seq:
            if 'compute' in a and t in comp_src:
                compute_overhead += profiler.cost_info[t]
                comp_src.remove(t)
        assert not comp_src, 'mdzz'
        total_overhead = memcpy_overhead + compute_overhead
        print(f'{100 * change_point:.0f}%\t{cost_new_plan:>8.0f}\t{memcpy_overhead:>8.0f}\t{compute_overhead:>8.0f}\t{total_overhead:>8.0f}\t'
              f'{100 * total_overhead / cost_new_plan:>6.2f}%\t{100 * (1 - total_overhead / cost_new_plan):>6.2f}%')
        adapt_results[change_point] = {
            'base': cost_new_plan,
            'ours.relayout': memcpy_overhead,
            'ours.recompute': compute_overhead,
            'ours.total': total_overhead
        }
    with open(os.path.join(data_dir, f'{model}_{batch}_{previous_budget_gb}-{new_budget_gb}.csv'), 'w') as f:
        f.write('finished,base,ours.relayout,ours.recompute,ours.total\n')
        for finished in (0.25, 0.5, 0.75):
            f.write(f'{finished},{adapt_results[finished]["base"]},{adapt_results[finished]["ours.relayout"]},'
                    f'{adapt_results[finished]["ours.recompute"]},{adapt_results[finished]["ours.total"]}\n')

    with open(os.path.join(data_dir, f'{model}_{batch}_{previous_budget_gb}-{new_budget_gb}.csv')) as f:
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 5))
        lines = [l for l in f]
        keys, baseline, ours_rp, ours_tot = [], [], [], []
        for line in lines[1:]:
            line = line.strip().split(',')
            base, memcpy, recompute, total = list(map(float, line[1:]))
            keys.append(line[0])
            baseline.append(1)
            ours_rp.append(recompute / base)
            ours_tot.append(total / base)

        plt.bar(x=range(3), height=baseline, width=bar_width, label='baseline', hatch='x', linewidth=lw, edgecolor='k')
        plt.bar(x=np.arange(3)+bar_width, height=ours_tot, width=bar_width, label='relayout', hatch='\\', linewidth=lw, edgecolor='k')
        plt.bar(x=np.arange(3)+bar_width, height=ours_rp, width=bar_width, label='recompute', hatch='/', linewidth=lw, edgecolor='k')
        plt.xticks(np.arange(3)+bar_width/2, keys, fontsize=30)
        plt.yticks(fontsize=30)
        ax.set_xlabel('Execution progress', fontsize=30)
        ax.set_ylabel('Adapting overhead', fontsize=30)
        # plt.legend(['Stop-restart', 'Ours (memcpy)', 'Ours (recompute)'], fontsize=20)
        plt.legend()
        plt.savefig(os.path.join(data_dir, f'{model}_{batch}_{previous_budget_gb}-{new_budget_gb}.pdf'), bbox_inches='tight', pad_inches=0)
    return adapt_results


if __name__ == '__main__':
    adaptiveness('MobilenetV2', 128, 8, 6)
    adaptiveness('MobilenetV2', 96, 8, 6)
    adaptiveness('Squeezenet', 128, 8, 6)
    adaptiveness('Squeezenet', 96, 8, 6)