#include "ExecutionPlanGenerator.hpp"
#include "SegmentTree.hpp"

string RECOMPUTE_SUFFIX = "_recompute";

string strip(string str, string trim = " ") {
    if (str.empty()) {
        return str;
    }
    str.erase(0, str.find_first_not_of(trim));
    str.erase(str.find_last_not_of(trim) + 1);
    return str;
}

vector<string> split(string str, string delimiter = "\\s+") {
    regex re(delimiter);
    vector<string> vec(sregex_token_iterator(str.begin(), str.end(), re, -1), sregex_token_iterator());
    return vec;

//    vector<string>tokens;
//    string::size_type lastPos = str.find_first_not_of(delimiter, 0);
//    string::size_type pos = str.find_first_of(delimiter, lastPos);
//    while (pos != string::npos || lastPos != string::npos) {
//        tokens.emplace_back(str.substr(lastPos, pos - lastPos));
//        lastPos = str.find_last_not_of(delimiter, pos);
//        pos = str.find_first_of(delimiter, lastPos);
//    }
//    return tokens;
}

int align_size(int size, int alignment = 64) {
    return (size + alignment - 1) / alignment * alignment;
}

Profiler::Profiler(string mn, int bs) : modelname(std::move(mn)), batchsize(bs) {
    string fileDir = "data/profiler/" + modelname + "/";
    bool load = true;
    string filenames[5] = {
            modelname + ".io_info.txt",
            modelname + "." + to_string(batchsize) + ".resize_info.txt",
            modelname + "." + to_string(batchsize) + ".cost_info.txt",
            modelname + "." + to_string(batchsize) + ".tensor_size.txt",
            modelname + "." + to_string(batchsize) + ".redundent_parent.txt",
    };
    for (auto i: filenames) {
        ifstream ifs(fileDir + i, ios::in);
        debug_print("%d: %s exist = %d\n", __LINE__, (fileDir + i).c_str(), ifs.good())
        load &= ifs.good();
        ifs.close();
    }
    if (load) {
        debug_print("%d: try load infos\n", __LINE__)
        load_infomation(fileDir);
    } else {
        debug_print("%d: try init from scratch\n", __LINE__)
        init_from_scratch();
    }
    set_thres_layers();
    dump_original_execution_info("heuristic/execution/" + modelname + "/" + modelname + ".execution.txt");
}

ostream &operator<<(ostream &out, Profiler::OpInfo &info) {
    out << "op(" << info.opid << ")\t";
    ostringstream oss;
    oss.str();
    copy(info.inputs.begin(), info.inputs.end(), ostream_iterator<string>(oss, ","));
    out << "inputs(" << oss.str() << ")\t";
    oss.clear();
    oss.str("");
    copy(info.outputs.begin(), info.outputs.end(), ostream_iterator<string>(oss, ","));
    out << "outputs(" << oss.str() << ")\t";
    oss.clear();
    oss.str("");
    copy(info.release.begin(), info.release.end(), ostream_iterator<string>(oss, ","));
    out << "release(" << oss.str() << ")";
    return out;
}

void Profiler::load_infomation(string fildDir) {
    load_io_info(fildDir + modelname + ".io_info.txt");
    debug_print("io_info.size() = %zu\n", io_info.size())
    load_resize_info(fildDir + modelname + "." + to_string(batchsize) + ".resize_info.txt");
    debug_print("resize_info.size() = %zu\n", resize_info.size())
    load_cost_info(fildDir + modelname + "." + to_string(batchsize) + ".cost_info.txt");
    debug_print("cost_info.size() = %zu\n", cost_info.size())
    load_tensor_size(fildDir + modelname + "." + to_string(batchsize) + ".tensor_size.txt");
    debug_print("tensor_size.size() = %zu\n", tensor_size.size())
    load_redundent_parent(fildDir + modelname + "." + to_string(batchsize) + ".redundent_parent.txt");
    debug_print("redundent_parent.size() = %zu\n", redundent_parent.size())
}

void Profiler::load_redundent_parent(string filename) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    string op, parent;
    while (infile >> op >> parent) {
        redundent_parent[op] = parent;
    }
    infile.close();
}

void Profiler::load_tensor_size(string filename) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    string op;
    size_t size;
    while (infile >> op >> size) {
        tensor_size[op] = size;
    }
    infile.close();
}

void Profiler::load_cost_info(string filename) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    string op;
    double cost;
    while (infile >> op >> cost) {
        cost_info[op] = cost;
    }
    infile.close();
}

void Profiler::load_resize_info(string filename) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    string line;
    while (getline(infile, line)) {
        vector<pair<string, string>> info;
        istringstream iss(line);
        int cnt;
        iss >> cnt;
        while (cnt--) {
            string action, opid;
            iss >> action >> opid;
            info.emplace_back(action, opid);
        }
        resize_info.push_back(info);
    }
    infile.close();
}

void Profiler::load_io_info(string filename) {
    ifstream infile(filename, ios::in);
    if (!infile.is_open()) {
        debug_print("%s: error to open %s\n", __FUNCTION__, filename.c_str())
        return;
    }
    string line;
    OpInfo info;
    while (getline(infile, line)) {
        istringstream iss(line);
        string tag, a;
        iss >> tag;
        if (tag == "opid") {
            info = OpInfo();
            iss >> info.opid;
        } else if (tag == "inputs") {
            while (iss >> a) {
                info.inputs.push_back(a);
            }
        } else if (tag == "outputs") {
            while (iss >> a) {
                info.outputs.push_back(a);
            }
        } else if (tag == "release") {
            while (iss >> a) {
                info.release.push_back(a);
            }
        } else if (tag == "temporary") {
            while (iss >> a) {
                info.temporary.push_back(a);
            }
        } else if (tag == "finish") {
            io_info.push_back(info);
//            debug_print("%s: %zu %zu %zu\n", info.opid.c_str(), info.inputs.size(), info.outputs.size(), info.release.size())
        } else {
            debug_print("invalid tag! choices are [inputs, outputs, release, temporary, finish]")
            return;
        }
    }
    infile.close();
}

void Profiler::set_thres_layers() {
    if (modelname == "Googlenet") {
        fp_thres = 1849;
        num_layers = -1;
    } else if (modelname == "MobilenetV2" || modelname == "MobilenetV2_CL") {
        fp_thres = 1750;
        num_layers = 30;
    } else if (modelname == "MobilenetV1") {
        num_layers = 31;
        fp_thres = 917;
    } else if (modelname == "Squeezenet" || modelname == "Squeezenet_CL") {
        fp_thres = 866;
        num_layers = 30;
    } else if (modelname == "Resnet50") {
        fp_thres = 1835;
        num_layers = 50;
    } else if (modelname == "MobilenetV1NoBN") {
        fp_thres = 121;
        num_layers = 31;
    } else if (modelname == "MobilenetV2NoBN") {
        fp_thres = 227;
        num_layers = 30;
    } else if (modelname == "SqueezenetNoBN") {
        fp_thres = 120;
        num_layers = 30;
    } else if (modelname == "Resnet50NoBN") {
        fp_thres = 298;
        num_layers = 50;
    }
}

void Profiler::init_from_scratch() {
    profile_from_scratch("profile/" + modelname + "/" + modelname + "." + to_string(batchsize) + ".profile.out");
    resize_from_scratch("resize/" + modelname + "/" + modelname + "." + to_string(batchsize) + ".resize.out");
    cost_from_scratch("cost/" + modelname + "/" + modelname + "." + to_string(batchsize) + ".cost.out");
    dump_information();
}

void Profiler::profile_from_scratch(string filename) {
    ifstream ifs(filename, std::ios::in);
    if (!ifs.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    string s;
    bool profile_flag = false;
    while (getline(ifs, s)) {
        if (s.find("start read-map") != string::npos) {
            profile_flag = true;
        } else if (s.find("finish read-map & start replace") != string::npos) {
            profile_flag = false;
        }
        if (!profile_flag) {
            continue;
        }

        if (strip(s, "\t").find("current Op") == 0) {
            io_info.emplace_back(OpInfo(to_string(io_info.size())));
        } else if (strip(s, "\t").find("outputs") == 0) {
            add_info(s, io_info[io_info.size() - 1].outputs);
        } else if (strip(s, "\t").find("inputs") == 0) {
            add_info(s, io_info[io_info.size() - 1].inputs);
        } else if (strip(s, "\t").find("release") == 0) {
            add_info(s, io_info[io_info.size() - 1].release);
        } else if (strip(s, "\t").find("temporary") == 0) {
            add_info(s, io_info[io_info.size() - 1].temporary);
        }
    }
    ifs.close();
    return;
}

void Profiler::resize_from_scratch(string filename) {
    ifstream ifs(filename, std::ios::in);
    if (!ifs.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    string s;
    bool resize_flag = false, compute_flag = false;
    int resize_tid = 0;
    string redundent = "", opid;
    set<string> freed;
    vector<pair<string, string>> current;
    int opidx = 0;

    while (getline(ifs, s)) {
        if (s.find("start read-map") != string::npos) {
            compute_flag = true;
        } else if (s.find("finish read-map & start replace") != string::npos) {
            compute_flag = false;
        }
        if (!compute_flag) {
            continue;
        }

        if (strip(s, "\t").find("current Op") == 0) {
            opid = to_string(opidx++);
            tensor_size[opid] = 0;
        }
        if (strip(s, "\t").find("finish allocate memory for cmd") == 0) {
            resize_flag = true;
            current.clear();
            freed.clear();
        }
        if (strip(s, "\t").find("try get") == 0) {
            if (resize_flag) {
                auto size = align_size(stoi(split(strip(s, "\t"))[2]));
                auto rtid = opid + ":" + to_string(resize_tid);
                tensor_size[rtid] = size;
                redundent = "";
                for (auto t: freed) {
                    if (tensor_size[t] == size) {
                        freed.erase(t);
                        redundent = t;
                        break;
                    }
                }
                if (!redundent.empty()) {
                    for (auto it = current.begin(); it != current.end();) {
                        if (it->second == redundent) {
                            it = current.erase(it);
                        } else {
                            it++;
                        }
                    }
                    redundent_parent[redundent] = rtid;
                }
                current.emplace_back(make_pair("alloc", rtid));
                resize_tid++;
            } else {
                tensor_size[opid] = align_size(stoi(split(strip(s, "\t"))[2]));
            }
        }
        if (strip(s, "\t").find("try return") == 0 && resize_flag) {
            auto size = align_size(stoi(split(strip(s, "\t"))[2]));
            vector<string> talloc;
            map<string, bool> tfree;
            for (auto p: current) {
                if (tensor_size[p.second] == size) {
                    if (p.first == "alloc") {
                        talloc.push_back(p.second);
                    } else {
                        tfree[p.second] = true;
                    }
                }
            }
            for (auto iter = talloc.rbegin(); iter != talloc.rend(); iter++) {
                if (!tfree[*iter]) {
                    current.emplace_back(make_pair("free", *iter));
                    freed.insert(*iter);
                    break;
                }
            }
        }
        if (strip(s, "\t").find("finish resize cmd") == 0) {
            resize_flag = false;
            resize_tid = 0;
            resize_info.push_back(current);
        }
    }
    ifs.close();
}

void Profiler::cost_from_scratch(string filename) {
    ifstream ifs(filename, std::ios::in);
    if (!ifs.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    string s;
    bool cost_flag = false;
    int opidx = 0;
    string op;
    while (getline(ifs, s)) {
        if (s.find("start read-map") != string::npos) {
            cost_flag = true;
        } else if (s.find("finish read-map & start replace") != string::npos) {
            cost_flag = false;
        }
        if (!cost_flag) {
            continue;
        }
        if (strip(s, "\t").find("current Op") == 0) {
            op = to_string(opidx++);
        } else if (s.find("cost time") != string::npos) {
            auto tmp = split(strip(s, "\t"));
            cost_info[op] = stod(tmp[tmp.size() - 2]);
        }
    }
    ifs.close();
}

void Profiler::add_info(string line, vector<string> &vec) {
    stringstream ss(line);
    char c;
    string tid, tsize;
    line = strip(split(line, ":")[1]);
    line = line.replace(line.find("["), 1, "").replace(line.find("]"), 1, "");
    auto items = split(line, ", ");
    set<string> tmp;
    for (auto item: items) {
        if (strip(item).size()) {
            auto tps = split(strip(item).replace(item.find("("), 1, "").replace(item.find(")"), 1, ""));
            tmp.insert(tps[0]);
            tensor_size[tps[0]] = stoi(tps[1]);
        }
    }
    vec.insert(vec.end(), tmp.begin(), tmp.end());
    return;
}

void Profiler::dump_information() {
    string fileDir = "data/profiler/" + modelname + "/";
    dump_io_info(fileDir + modelname + ".io_info.txt");
    dump_resize_info(fileDir + modelname + "." + to_string(batchsize) + ".resize_info.txt");
    dump_cost_info(fileDir + modelname + "." + to_string(batchsize) + ".cost_info.txt");
    dump_tensor_size(fileDir + modelname + "." + to_string(batchsize) + ".tensor_size.txt");
    dump_redundent_parent(fileDir + modelname + "." + to_string(batchsize) + ".redundent_parent.txt");
}

void Profiler::dump_io_info(string filename) {
    ofstream ofs(filename, ios::out);  // io_info
    if (!ofs.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    for (auto info: io_info) {
        ofs << "opid " << info.opid << "\n";
        ofs << "inputs ";
        for (auto i: info.inputs) {
            ofs << i << " ";
        }
        ofs << "\n";

        ofs << "outputs ";
        for (auto i: info.outputs) {
            ofs << i << " ";
        }
        ofs << "\n";

        ofs << "release ";
        for (auto i: info.release) {
            ofs << i << " ";
        }
        ofs << "\n";

        ofs << "temporary ";
        for (auto i: info.temporary) {
            ofs << i << " ";
        }
        ofs << "\n";

        ofs << "finish\n";
    }
    ofs.close();
}

void Profiler::dump_resize_info(string filename) {
    ofstream ofs(filename, ios::out);  // io_info
    if (!ofs.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    for (auto info: resize_info) {
        ofs << info.size() << " ";
        for (auto i: info) {
            ofs << i.first << " " << i.second << " ";
        }
        ofs << "\n";
    }
    ofs.close();
}

void Profiler::dump_cost_info(string filename) {
    ofstream ofs(filename, ios::out);
    if (!ofs.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    for (auto info: cost_info) {
        ofs << info.first << " " << info.second << "\n";
    }
    ofs.close();
}

void Profiler::dump_tensor_size(string filename) {
    ofstream ofs(filename, ios::out);
    if (!ofs.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    for (auto info: tensor_size) {
        ofs << info.first << " " << info.second << "\n";
    }
    ofs.close();
}

void Profiler::dump_redundent_parent(string filename) {
    ofstream ofs(filename, ios::out);
    if (!ofs.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    for (auto info: redundent_parent) {
        ofs << info.first << " " << info.second << "\n";
    }
    ofs.close();
}

void Profiler::dump_original_execution_info(string filename) {
    ofstream ofs(filename, ios::out);
    if (!ofs.is_open()) {
        debug_print("error to %s\n", __FUNCTION__)
        return;
    }
    for (auto info: io_info) {
        ofs << "compute" << " " << info.opid << "\n";
        for (auto t: info.release) {
            ofs << "release" << " " << t << "\n";
        }
    }
    ofs.close();
}


Recomputer::Recomputer(shared_ptr<Profiler> profiler_ptr, shared_ptr<GreedyAllocator> allocator_ptr, size_t budget, double thres)
        : profiler(profiler_ptr), grd_allocator(allocator_ptr), budget_mb(budget), threshold(thres) {
    vnl_allocator = VanillaAllocator();
    // A set tracking which tensors are currently "in-memory"
    allocated_tensor.clear();
    // A set of tensors that are candidates for being evicted and recomputed
    feature_map.clear();
    debug_print("mem_bgt = %zu, budget_mb = %zu\n", budget_b, budget_mb)
    for (int i = 0; i < profiler->io_info.size(); i++) {
        for (auto t : profiler->io_info[i].inputs) {
            table_in[to_string(i)].insert(t);
            table_out[t].insert(to_string(i));
        }
    }
    // info.release contains the IDs of tensors that are obsolete at current step.
    // The map release_point is a map[k, v] that tensor k should be release at tensor v
    for (auto &info: profiler_ptr->io_info) {
        for (auto t: info.release) {
            release_point[t] = info.opid;
        }
    }
    // Tensors that are never explicitly released will be released at the final step
    for (int i = 0; i < profiler->io_info.size(); i++) {
        if (release_point.find(to_string(i)) == release_point.end()) {
            release_point[to_string(i)] = to_string(profiler->io_info.size());
        }
    }
    computing_budget["MobilenetV1:5500"] = {
            {32,  5000},
            {64,  3000},
            {96,  2600},
            {128, 2100},
    };
    computing_budget["Resnet50:5500"] = {
            {32, 4800},
            {64, 3900},
            {96, 3100},
    };
    computing_budget["MobilenetV2:5500"] = {
            {40,  5000},
            {48,  3900},
            {64,  4500},
            {80,  4100},
            {96,  3900},
            {112, 4400},
            {128, 2700},
            {144, 3000},
            {160, 3200},
            {176, 3100},
            {192, 3200},
            {208, 2000},
    };
    computing_budget["Squeezenet:5500"] = {
            {48,  4000},
            {64,  4400},
            {80,  4600},
            {96,  3900},
            {112, 3500},
            {128, 3600},
            {144, 3000},
            {160, 2600},
            {176, 2700},
            {192, 2300},
    };
    computing_budget["MobilenetV2:3800"] = {
            {40,  3500},
            {48,  3500},
            {64,  2900},
            {80,  2400},
            {96,  2100},
            {112, 2200},
            {128, 2200},
    };
    computing_budget["Squeezenet:3800"] = {
            {48,  3300},
            {64,  2600},
            {80,  2400},
            {96,  2000},
            {112, 1700},
            {128, 1400},
    };
    budget_b = computing_budget[profiler->modelname + ":" + to_string(budget_mb)][profiler->batchsize] << 20;
    debug_print("budget_b = %lu\n", budget_b)
}

void Recomputer::ondemand_recompute_via_metric() {
    get_feature_map();
    ostringstream oss;
    oss.str();
    copy(feature_map.begin(), feature_map.end(), ostream_iterator<string>(oss, ","));
    debug_print("feature_map(%s)\n", oss.str().c_str())
    oss.clear();
    oss.str("");
    debug_print("profiler.io_info.size() = %zu\n", profiler->io_info.size())
    for (auto op = 0; op < profiler->io_info.size(); op++) {
        if (op % 100 == 0) {
            debug_print("%s: current op is [%4d/%lu], exe_seq.size() = %zu\n", __FUNCTION__, op, profiler->io_info.size(), exe_seq.size())
        }
//        cout << profiler->io_info[op];
        debug_print("\tcur_size:%zu\n", vnl_allocator.current_size())
        auto &info = profiler->io_info[op];
        auto comp_src = get_compute_source(info.opid);
        if (!comp_src.empty()) {
            vector<string> src(comp_src.begin(), comp_src.end());
            sort(src.begin(), src.end(), [](string a, string b) { return stoi(a) < stoi(b); });
            for (auto i : src) {
                ondemand_compute(i, true, table_in[info.opid]);
            }
        }
        auto skip_rel = op + 1 < profiler->io_info.size() ? table_in[to_string(op + 1)] : set<string>();
        ondemand_compute(info.opid, false, skip_rel);
        current_progress = info.opid;
//        debug_print("allocator.cur = %zu\n", allocator.current_size())
    }
//    return;
    debug_print("%s: exe_seq.size = %lu\n", __FUNCTION__, exe_seq.size())
    adjust_exe_seq();
    debug_print("finish decision making\n")
    dump_exe_seq();
    debug_print("model=%s\tbatch=%d\tbudget=%lu\texe_seq.size()=%lu\n",
                profiler->modelname.c_str(), profiler->batchsize, computing_budget[profiler->modelname + ":" + to_string(budget_mb)][profiler->batchsize], exe_seq.size())
}

void Recomputer::update_metric(string ith) {
    auto comp_src = get_compute_source(ith);
    comp_src.insert(ith);
    double comp_t = 0;
    for (auto src : comp_src) {
        comp_t += profiler->cost_info[src];
    }
//    debug_print("%s.comp_t=%.3lf\n", ith.c_str(), comp_t)
    metric[ith] = profiler->tensor_size[ith] * (stod(release_point[ith]) - stod(current_progress)) / comp_t;
//    debug_print("%d - %d = %.3lf --> %.3lf\n", stoi(release_point[ith]), stoi(current_progress), (stod(release_point[ith]) - stod(current_progress)), metric[ith]);
//    metric[ith] = profiler->tensor_size[ith]  / comp_t;
}

void Recomputer::ondemand_compute(string ith, bool recompute, set<string> skip_pre_rel) {
    for (auto t : profiler->io_info[stoi(ith)].outputs) {
        vnl_allocator.alloc(ith, profiler->tensor_size[t]);
        allocated_tensor.insert(t);
    }
    for (auto p : profiler->resize_info[stoi(ith)]) {
        if (p.first == "alloc") {
            vnl_allocator.alloc(p.second, profiler->tensor_size[p.second]);
        }
    }
    exe_seq.emplace_back(recompute ? "recompute" : "compute", ith);
    debug_print("exe_seq[%zu]=(%s, %s)\n", exe_seq.size() - 1, exe_seq[exe_seq.size() - 1].first.c_str(), exe_seq[exe_seq.size() - 1].second.c_str())
    for (auto p : profiler->resize_info[stoi(ith)]) {
        if (p.first == "free") {
            vnl_allocator.free(p.second);
        }
    }
    ostringstream oss;
    auto &rel = profiler->io_info[stoi(ith)].release;
    copy(rel.begin(), rel.end(), ostream_iterator<string>(oss, ","));
//    cout<<"release: "<<oss.str()<<endl;
    for (auto t : profiler->io_info[stoi(ith)].release) {
//        debug_print("%s: try release %s\n", ith.c_str(), t.c_str())
        vnl_allocator.free(t);
        if (allocated_tensor.find(t) == allocated_tensor.end()) {
            debug_print("fuck %s\n", t.c_str())
        } else {
            allocated_tensor.erase(t);
            exe_seq.emplace_back(recompute ? "rerelease" : "release", t);
            debug_print("exe_seq[%zu]=(%s, %s)\n", exe_seq.size() - 1, exe_seq[exe_seq.size() - 1].first.c_str(), exe_seq[exe_seq.size() - 1].second.c_str())
        }
    }
    debug_print("cur_size(%zu)\tbudget_MB=%lu\tbudget_b*thres(%.0lf)\n",
                vnl_allocator.current_size(), budget_b >> 20, budget_b * threshold)
    while (vnl_allocator.current_size() > budget_b * threshold) {
        // debug_print("%s\t%zu\n", ith.c_str(), allocator.current_size());
        set<string> comped_fms, _tmp;
        set_intersection(feature_map.begin(), feature_map.end(),
                         allocated_tensor.begin(), allocated_tensor.end(),
                         inserter(_tmp, _tmp.begin()));
        set_difference(_tmp.begin(), _tmp.end(), skip_pre_rel.begin(), skip_pre_rel.end(), inserter(comped_fms, comped_fms.begin()));
        if (comped_fms.empty()) {
            break;
        }
        string evict_t;
        for (auto t : comped_fms) {
            update_metric(t);
            debug_print("metric_tps[%s]=%.3lf\t", t.c_str(), metric[t])
            if (evict_t.empty() || metric[evict_t] < metric[t]) {
                evict_t = t;
            }
        }
        debug_print("\n")
        if (evict_t.empty()) {
            break;
        }
        exe_seq.emplace_back("pre-release", evict_t);
        debug_print("exe_seq[%zu]=(%s, %s)\n", exe_seq.size() - 1, exe_seq[exe_seq.size() - 1].first.c_str(), exe_seq[exe_seq.size() - 1].second.c_str())
        allocated_tensor.erase(evict_t);
        vnl_allocator.free(evict_t);
    }
}

set<string> Recomputer::get_compute_source(string ith) {
    set<string> comp_src;
    queue<string> que;
    for (auto tid : table_in[ith]) {
        if (allocated_tensor.find(tid) == allocated_tensor.end() && allocated_tensor.find(tid + RECOMPUTE_SUFFIX) == allocated_tensor.end()) {
            que.push(tid);
        }
    }
    while (!que.empty()) {
        auto cur = que.front();
        que.pop();
        if (comp_src.find(cur) == comp_src.end() && allocated_tensor.find(cur) == allocated_tensor.end()) {
            comp_src.insert(cur);
            for (auto t : table_in[cur]) {
                que.push(t);
            }
        }
    }
    return comp_src;
}

void Recomputer::adjust_exe_seq() {
    debug_print("before %s: exe_seq.size() = %lu\n", __FUNCTION__, exe_seq.size())
    vector<int> to_adjust;
    debug_print("%s: exe_seq %lu\n", __FUNCTION__, exe_seq.size())
    for (int i = 0; i < exe_seq.size(); i++) {
        if (exe_seq[i].first.find("release") != string::npos) {
            to_adjust.push_back(i);
//            debug_print("to_adj: %d\n", i);
        }
    }
    debug_print("%s: to_adjust %lu\n", __FUNCTION__, to_adjust.size())
    for (auto idx : to_adjust) {
        auto p = exe_seq[idx];
        auto pos = -1;
        for (auto i = idx; i >= 0; i--) {
            auto &inputs = profiler->io_info[stoi(exe_seq[i].second)].inputs;
            if (exe_seq[i].first.find("compute") != string::npos &&
                (exe_seq[i].second == p.second || find(inputs.begin(), inputs.end(), p.second) != inputs.end())) {
                pos = i + 1;
                break;
            }
        }
        if (idx != pos) {
//            debug_print("adjust: %d to %d\n", idx, pos)
            exe_seq.erase(exe_seq.begin() + idx);
            exe_seq.insert(exe_seq.begin() + pos, p);
//            debug_print("after adjust: exe_seq[%d] = (%s, %s)\n", pos, p.first.c_str(), p.second.c_str())

        }
    }
    debug_print("after %s: exe_seq.size() = %lu\n", __FUNCTION__, exe_seq.size())
}

void Recomputer::get_feature_map() {
    for (int i = 0; i < profiler->fp_thres; i++) {
        for (auto t : profiler->io_info[i].outputs) {
            feature_map.insert(t);
        }
        for (auto t : profiler->io_info[i].release) {
            feature_map.erase(t);
        }
    }
}

void Recomputer::calibrated_compute(string ith, bool recompute) {
    auto current_size = grd_allocator->current_size(calibrated_timestamp);
    while (grd_allocator->current_size(calibrated_timestamp) > budget_b) {
        string evict_t;
        for (auto t: allocated_tensor) {
            // `metric` is a member vairable (map<string, double>) that stores
            // the TPS score for each tensor
            update_metric(t);
            debug_print("metric_tps[%s]=%.3lf\n", t.c_str(), metric[t])
            // find the tensor with the highest score
            if (evict_t.empty() || metric[evict_t] < metric[t]) {
                evict_t = t;
            }
        }
        if (evict_t.empty()) {
            debug_print("cannot evict any tensor because current allocated-tensor set is empty")
            break;
        }
        grd_allocator->remove_tensor(evict_t, calibrated_timestamp);
        exe_seq.emplace_back("pre-release", evict_t);
        debug_print("exe_seq[%zu]=(%s, %s)\n", exe_seq.size() - 1, exe_seq[exe_seq.size() - 1].first.c_str(), exe_seq[exe_seq.size() - 1].second.c_str())
        allocated_tensor.erase(evict_t);
    }
    // do NOT add resize-tensor to allocated-tensor because its cost is N/A
    // during the recompute process, all tensor are considered equally but the reisze-tensor is only a tmp var
    // it is only used to assist the computation of the other tensors, i.e., the output of each OP
    for (auto t : profiler->io_info[stoi(ith)].outputs) {
        grd_allocator->allocate(t);
        // it's important to allocate to know the compute seq
        allocated_tensor.insert(t);
    }
    exe_seq.emplace_back(recompute ? "recompute" : "compute", ith);
    debug_print("exe_seq[%zu]=(%s, %s)\n", exe_seq.size() - 1, exe_seq[exe_seq.size() - 1].first.c_str(), exe_seq[exe_seq.size() - 1].second.c_str())
    for (auto t : profiler->io_info[stoi(ith)].release) {
        if (allocated_tensor.find(t) == allocated_tensor.end()) {
            debug_print("%s: fuck %s\n", __FUNCTION__, t.c_str())
        } else {
            allocated_tensor.erase(t);
            exe_seq.emplace_back(recompute ? "re-release" : "release", t);
            debug_print("exe_seq[%zu]=(%s, %s)\n", exe_seq.size() - 1, exe_seq[exe_seq.size() - 1].first.c_str(), exe_seq[exe_seq.size() - 1].second.c_str())
        }
    }

    calibrated_timestamp++;
}

void Recomputer::memory_calibrated_progressive_recomputation() {
    debug_print("profiler.io_info.size() = %zu\n", profiler->io_info.size())
    for (auto opidx = 0; opidx < profiler->io_info.size(); ++opidx) {
        if (opidx % 100 == 0) {
            debug_print("%s: current op is [%4d/%lu], exe_seq.size() = %zu\n",
                        __FUNCTION__, opidx, profiler->io_info.size(), exe_seq.size())
        }
        auto &info = profiler->io_info[opidx];

        // comp_src will contains missing input operators that must be recomputed
        auto comp_src = get_compute_source(info.opid);

        if (!comp_src.empty()) {
            // there are missing inputs; recompute them
            vector<string> src(comp_src.begin(), comp_src.end());

            // sort so that the missing tensors will be computed in the correct dependency order
            sort(src.begin(), src.end(), [](string a, string b) { return stoi(a) < stoi(b); });

            // "stretches" the lifetime of existing tensors to account for the additional recomputation
            grd_allocator->insert_tensors(src, calibrated_timestamp);
            for(auto tid: src) {
                calibrated_compute(tid, true);
            }
        }
        calibrated_compute(info.opid, false);
        current_progress = info.opid;
    }
    debug_print("%s: exe_seq.size = %lu\n", __FUNCTION__, exe_seq.size())
    adjust_exe_seq();
    debug_print("finish decision making\n")
    dump_exe_seq();
    debug_print("model=%s\tbatch=%d\tbudget=%lu\texe_seq.size()=%lu\n",
                profiler->modelname.c_str(), profiler->batchsize, computing_budget[profiler->modelname + ":" + to_string(budget_mb)][profiler->batchsize], exe_seq.size())
}

void Recomputer::dump_exe_seq() {
    string filename = "heuristic/execution/" + profiler->modelname + "/" +
                      profiler->modelname + "." + to_string(profiler->batchsize) + "." + to_string(budget_mb) + ".execution.txt";
    debug_print("%s: filename = %s\n", __FUNCTION__, filename.c_str())
    ofstream ofs(filename, ofstream::out);
    if (ofs.is_open()) {
        debug_print("begin %s\n", __FUNCTION__)
        for (auto &p : exe_seq) {
            ofs << p.first << "\t" << p.second << endl;
        }
        ofs.close();
    } else {
        debug_print("%s: error to dump result to %s!\n", __FUNCTION__, filename.c_str())
    }
    debug_print("%s: finish\n", __FUNCTION__)
    ofs.close();
}

GreedyAllocator::GreedyAllocator(shared_ptr<Profiler> profiler_ptr, size_t bgt_mb, bool norecomp)
        : profiler(std::move(profiler_ptr)), budget_mb(bgt_mb), noRecompute(norecomp) {}


struct GreedyAllocator::Tensor {
    string id;
    int free, alloc, life;  // lifetime is [alloc, free)
    size_t size;

    Tensor(string i = "", int a = -1, int f = -1, size_t s = 0) : id(i), free(f), alloc(a), size(s) {
        life = free - alloc;
    }

    bool operator<(const Tensor &b) const {
        if (life != b.life) {
            return life > b.life;   // sort by lifetime, descending
        } else if (size != b.size) {
            return size > b.size;   // sort by size, descending
        } else if (alloc != b.alloc) {
            return alloc < b.alloc; // sort by allocation time, ascending
        } else {
            return free < b.free;   // sort by free time, ascending
        }
    }
};

bool GreedyAllocator::load_info_via_exe_seq() {
    string ifilename = "heuristic/execution/" + profiler->modelname + "/" +
                       profiler->modelname + "." + to_string(profiler->batchsize) + "." + to_string(budget_mb) + ".execution.txt";
    string ofilename = "data/heu_info/" + profiler->modelname + "/" +
                       profiler->modelname + "." + to_string(profiler->batchsize) + "." + to_string(budget_mb) + ".heu_info.txt";
    if (noRecompute) {
        ifilename = "heuristic/execution/" + profiler->modelname + "/" +
                    profiler->modelname + ".execution.txt";
        ofilename = "data/heu_info/" + profiler->modelname + "/" +
                    profiler->modelname + "." + to_string(profiler->batchsize) + ".heu_info.txt";
    }
    ifstream ifs(ifilename);
    if (!ifs.is_open()) {
        debug_print("%s: error to load exe_seq file %s\n", __FUNCTION__, ifilename.c_str())
        return false;
    }
    map<string, int> op_comp_idx;
    int idx = -1;
    vector<Tensor> recomp_seq_info;
    string a, op;
    vector<pair<string, string>> update_redundent_parent;
    map<string, shared_ptr<Tensor>> heu_info;
    while (ifs >> a >> op) {
        if (a.find("compute") != string::npos) {
            op_comp_idx[op] = ++idx;
            recomp_seq_info.emplace_back(op, idx);
            if (a == "recompute") {
                for (const auto &iter : profiler->redundent_parent) {
                    auto pos = iter.first.find(":");
                    if (pos != string::npos) {
                        string redundent = to_string(idx) + ":" + iter.first.substr(pos + 1);
                        update_redundent_parent.emplace_back(redundent, iter.second);
                    }
                }
            }
        } else {
            auto iter = op_comp_idx.find(op);
            recomp_seq_info[iter->second].free = recomp_seq_info.size();
            op_comp_idx.erase(iter);
        }
    }
    for (auto &p : update_redundent_parent) {
        profiler->redundent_parent[p.first] = p.second;
    }
    update_redundent_parent.clear();
    for (const auto &iter : op_comp_idx) {
        recomp_seq_info[iter.second].free = recomp_seq_info.size();
    }
    op_comp_idx.clear();
    for (int i = 0; i < recomp_seq_info.size(); i++) {
        // output info
        auto &info = recomp_seq_info[i];
        heu_info[to_string(i)] = make_shared<Tensor>(to_string(i), info.alloc, info.free, profiler->tensor_size[info.id]);
        // resize info
        for (auto p : profiler->resize_info[stoi(info.id)]) {
            if (p.first == "alloc") {
                string rsid = to_string(i) + ":" + p.second.substr(p.second.find(':') + 1);
                heu_info[rsid] = make_shared<Tensor>(rsid, i, i + 1, profiler->tensor_size[p.second]);
            }
        }
    }
    ofstream ofs(ofilename);
    if (!ofs.is_open()) {
        debug_print("error to dump infos to %s\n", ofilename.c_str())
    } else {
        int maxf = 0;
        for (auto p : heu_info) {
            infos.push_back(p.second);
            id2tensor[p.second->id] = p.second;
            maxf = max(maxf, p.second->free);
            // i >> a >> f >> s
            ofs << p.second->id << " " << p.second->alloc << " " << p.second->free << " " << p.second->size << endl;
        }
    }
    ifs.close();
    ofs.close();
    id2originalTensor = id2tensor;  // copy-assignment
    return true;
}

bool GreedyAllocator::overlap(shared_ptr<Tensor> t1, shared_ptr<Tensor> t2) {
    return !(t1->free <= t2->alloc || t2->free <= t1->alloc);
}

bool GreedyAllocator::overlap(MemoryAddress m1, MemoryAddress m2) {
    return !(m1.second <= m2.first || m2.second <= m1.first);
}

bool GreedyAllocator::mergeable(MemoryAddress m1, MemoryAddress m2) {
    return !(m1.second < m2.first || m2.second < m1.first);
}

/// A sweep-line algorithm to populate the `up_tensors` and `down_tensors` maps
void GreedyAllocator::build_topology() {
  vector<vector<shared_ptr<Tensor>>> allocated_tensors_each_timestamp,
      freed_tensors_each_timestamp;
  sort(infos.begin(), infos.end(),
       [this](shared_ptr<Tensor> p, shared_ptr<Tensor> q) {
         if (p->alloc != q->alloc) {
           return p->alloc < q->alloc;
         }
         return tensor2address[p].first < tensor2address[q].first;
       });
  int max_time = 0;
  for (auto t : infos) {
    max_time = max(max_time, t->free);
  }
  allocated_tensors_each_timestamp.resize(max_time + 1);
  freed_tensors_each_timestamp.resize(max_time + 1);
  for (auto t : infos) {
    allocated_tensors_each_timestamp[t->alloc].push_back(t);
    freed_tensors_each_timestamp[t->free].push_back(t);
  }
  vector<pair<shared_ptr<Tensor>, MemoryAddress>> scan_line;
  for (int timestamp = 0; timestamp < allocated_tensors_each_timestamp.size();
       timestamp++) {
    for (auto t : freed_tensors_each_timestamp[timestamp]) {
      auto pos = find(scan_line.begin(), scan_line.end(),
                      make_pair(t, tensor2address[t]));
      scan_line.erase(pos);
    }
    for (auto t : allocated_tensors_each_timestamp[timestamp]) {
      auto pos = lower_bound(scan_line.begin(), scan_line.end(),
                             make_pair(t, tensor2address[t]),
                             [](pair<shared_ptr<Tensor>, MemoryAddress> p,
                                pair<shared_ptr<Tensor>, MemoryAddress> q) {
                               return p.second < q.second;
                             }) -
                 scan_line.begin();
      scan_line.insert(pos + scan_line.begin(),
                       make_pair(t, tensor2address[t]));
      if (pos + 1 < scan_line.size()) {
        up_tensors[t].push_back(scan_line[pos + 1].first);
        down_tensors[scan_line[pos + 1].first].push_back(t);
      }
      if (pos - 1 >= 0) {
        up_tensors[scan_line[pos - 1].first].push_back(t);
        down_tensors[t].push_back(scan_line[pos - 1].first);
      }
    }
  }
}

void GreedyAllocator::check_topology() {
    // map<shared_ptr<Tensor>, vector<shared_ptr<Tensor>>> up_tensors, down_tensors;

    // if B is above A, make sure A is in B's down list
    for (auto const& a: up_tensors) {
        auto t_a = a.first;
        auto tensors_above_a = a.second;
        for (auto const& t_b: tensors_above_a) {
            auto tensors_below_b = down_tensors[t_b];
            if (find(tensors_below_b.begin(), tensors_below_b.end(), t_a) == tensors_below_b.end()) {
                debug_print("!!!!!!!! invalid toplogy: %s is above %s, but %s is not in %s's down list",
                        t_b->id.c_str(), t_a->id.c_str(), t_a->id.c_str(), t_b->id.c_str());
            }
        }
    }

    for (auto const& a: down_tensors) {
        auto t_a = a.first;
        auto tensors_below_a = a.second;
        for (auto const& t_b: tensors_below_a) {
            auto tensors_above_b = up_tensors[t_b];
            if (find(tensors_above_b.begin(), tensors_above_b.end(), t_a) == tensors_above_b.end()) {
                debug_print("!!!!!!!! invalid toplogy: %s is below %s, but %s is not in %s's above list",
                        t_b->id.c_str(), t_a->id.c_str(), t_a->id.c_str(), t_b->id.c_str());
            }
        }
    }
}

void GreedyAllocator::remove_tensor(string tid, int timestamp) {
    // (Jinyang)
    // remove the tensor `tid` from the memory allocation plan at `timestamp`
    //
    // The removal triggers a cascade of adjustments of the memory layout:
    // tensors that were "above" the removed tensor can now "sink" into the
    // freed space. This function recalculated the positions of these affected
    // tensors.

    // find the tensors upper to the removed tensor
    auto tensor = id2tensor[tid];
    map<shared_ptr<Tensor>, bool> visited;
    queue<shared_ptr<Tensor>> que;
    vector<shared_ptr<Tensor>> influenced_tensors;

    // (Jinyang)
    // Below code is bugged: it put the tensor to be removed in the
    // influenced_tensors.
    //
    // que.push(tensor);
    // visited[tensor] = true;
    //
    // This is replaced by the following code:
    for (auto t: up_tensors[tensor]) {
        if (!visited[t]) {
            visited[t] = true;
            que.push(t);
        }
    }
    // (Jinyang)
    // tensors above `tensor` are "influenced" (affected)
    // find them through BFS search
    while (!que.empty()) {
        auto top = que.front();
        que.pop();
        influenced_tensors.push_back(top);
        for (auto t: up_tensors[top]) {
            if (!visited[t]) {
                visited[t] = true;
                que.push(t);
            }
        }
    }
    // the tensors that are above it can sink down, remove current tensor and not iter++
    vector<shared_ptr<Tensor>> remained_tensors;

    // (Jinyang)
    // a segment tree is used to store the highest address that the tensors
    // above `tensor` can sink within `tensor`'s lifetime
    //
    // this is a max segment tree (greater<size_t>):
    //
    // array = [-inf, -inf, -inf, ..., -inf]
    //          ^tensor->alloc         ^tensor->free
    SegmentTree<size_t, greater<size_t>> segmentTree(tensor->alloc, tensor->free);

    // (Jinyang)
    // tensor2address[tensor] returns a MemoryAddress which is pair: [start, end)
    // So current_top is the starting mem addr of the removed tensor, i.e., the
    // floor tensors above should sink to
    //
    // [current_top, current_top, ..., current_top]
    //  ^alloc                         ^free
    size_t current_top = tensor2address[tensor].first;
    for (int i = tensor->alloc; i < tensor->free; i++) {
        segmentTree.insert(i, current_top);
    }

    // (Jinyang)
    // remove the tensor
    tensor2address.erase(tensor);

    for (auto t: influenced_tensors) {
        // influenced tensor is order by the position  relative to removed tensor, [0] is the downmost one
        if (t->alloc >= tensor->alloc && t->free <= tensor->free) {
            // (Jinyang)
            // this is the easy case:
            // t's lifetime is entirely contained within tensor's lifetime
            // if tensor is removed, t can sink down during its lifetime

            // naive sink down is not okay
//            tensor2address[t].first -= tensor->size;
//            tensor2address[t].second -= tensor->size;
            // NOW: sinking down efficiently through segment tree
            current_top = segmentTree.query(t->alloc, t->free - 1);
            tensor2address[t].first = current_top;
            tensor2address[t].second = current_top + t->size;
            segmentTree.insert(make_pair(t->alloc, t->free -1), current_top + t->size);
        } else {
            // (Jinyang)
            // this is the more tricky case:
            // there is overlap between t's and tensor's lifetime, but not
            // fully, and we will re-allocate them later
            remained_tensors.push_back(t);
        }
    }
    // (Jinyang)
    // We are not deleteing the entire `tensor`, only [timestamp, tensor->free).
    // We keep [tensor->alloc, timestamp) as a new tensor_left_part which will
    // be allocated later

    // the left part (splitted at `timestamp`) remains
    auto tensor_left_part = make_shared<Tensor>(tensor->id, tensor->alloc, timestamp, tensor->size);
    remained_tensors.push_back(tensor_left_part);

    // rebuild the topology of graph
    // First, connect the neighbors of the tensor being removed.
    auto down_neighbors = down_tensors[tensor];
    auto up_neighbors = up_tensors[tensor];
    for (auto& t_down : down_neighbors) {
        for (auto& t_up : up_neighbors) {
            if (overlap(t_down, t_up)) {
                // Safely add t_up to t_down's up-list.
                auto it_down = up_tensors.find(t_down);
                if (it_down != up_tensors.end()) {
                    auto& up_list = it_down->second;
                    if (find(up_list.begin(), up_list.end(), t_up) == up_list.end()) {
                        up_list.push_back(t_up);
                    }
                }

                // Safely add t_down to t_up's down-list.
                auto it_up = down_tensors.find(t_up);
                if (it_up != down_tensors.end()) {
                    auto& down_list = it_up->second;
                    if (find(down_list.begin(), down_list.end(), t_down) == down_list.end()) {
                        down_list.push_back(t_down);
                    }
                }
            }
        }
    }

    // Now, safely detach the tensor itself.
    for (auto t: down_neighbors) {
        auto it = up_tensors.find(t);
        if (it != up_tensors.end()) {
            auto& up_list = it->second;
            up_list.erase(remove(up_list.begin(), up_list.end(), tensor), up_list.end());
        }
    }
    for (auto t: up_neighbors) {
        auto it = down_tensors.find(t);
        if (it != down_tensors.end()) {
            auto& down_list = it->second;
            down_list.erase(remove(down_list.begin(), down_list.end(), tensor), down_list.end());
        }
    }
    down_tensors.erase(tensor);
    up_tensors.erase(tensor);  //
    // removed tensor's id now points to the left part of the tensor, update all infomation
    auto it = std::find_if(infos.begin(), infos.end(), 
                           [&](const shared_ptr<Tensor>& p){ return p->id == tensor->id; });
    if (it != infos.end()) {
        infos.erase(it);
    }
    infos.push_back(tensor_left_part);
    id2tensor[tensor->id] = tensor_left_part;
    // remove the remained tensors from the allocated tensors, because their address need to be updated
    for (auto t: remained_tensors) {
        tensor2address.erase(t);
    }

    //insert the remained_tensors (including the left part of the removed tensor) to the pool
    sort(remained_tensors.begin(), remained_tensors.end());
    for (auto t: remained_tensors) {
        // cannot use `get_best_address` because iter iterates tensor2address, which points to a pair not a Tensor
        vector<MemoryAddress> madd;

        // (Jinyang)
        // We are trying to alloc t. This for loop checks if its lifetime
        // overlaps with any allocated tensors
        for (auto iter: tensor2address) {
            if (overlap(t, iter.first)) { // the lifetime
                auto pos = lower_bound(madd.begin(), madd.end(), iter.second) - madd.begin();
                if (pos != 0 && mergeable(madd[pos - 1], iter.second)) {
                    // (Jinyang) mergeable: overlap or adjacent
                    madd[pos - 1].first = min(madd[pos - 1].first, iter.second.first);
                    madd[pos - 1].second = max(madd[pos - 1].second, iter.second.second);
                    pos--;
                } else {
                    madd.insert(madd.begin() + pos, iter.second);
                }
                while (pos + 1 < madd.size() && mergeable(madd[pos], madd[pos + 1])) {
                    madd[pos].first = min(madd[pos].first, madd[pos + 1].first);
                    madd[pos].second = max(madd[pos].second, madd[pos + 1].second);
                    madd.erase(madd.begin() + pos + 1);
                }
            }
        }
        if (madd.empty()) {
            madd.emplace_back(0, 0);
        }
        MemoryAddress best(-1, -1);
        // (Jinyang)
        // Handle case where the best spot is the bottom of the address space
        // madd[0] is the first occupied block and madd[0].first is its starting address
        if (madd[0].first >= t->size && (best.first == -1 || madd[0].first < best.second - best.first)) {
            best = MemoryAddress(0, madd[0].first);
        }

        // (Jinyang)
        // Find the best-fit gap: smallest large-enough gap that fits t
        for (int iter = 0; iter + 1 < madd.size(); iter++) {
            if (madd[iter + 1].first - madd[iter].second >= t->size &&
                (best.first == -1 || madd[iter + 1].first - madd[iter].second < best.second - best.first)) {
                best = MemoryAddress(madd[iter].second, madd[iter + 1].first);
            }
        }

        // (Jinyang)
        // No gap was found and we need to alloc t at the top of the pool
        if (best.first == -1) {
            best = MemoryAddress(madd[madd.size() - 1].second, numeric_limits<size_t>::max());
        }
        tensor2address[t] = make_pair(best.first, best.first + t->size);
    }
}


void GreedyAllocator::extend_pool(int timestamp, int length) {
    // [timestame, timestame + length)
    for (auto t: infos) {
        if (timestamp <= t->alloc) {
            t->alloc += length;
            t->free += length;
        } else if (timestamp < t->free) {
            t->free += length;
        }
    }

}

void GreedyAllocator::insert_tensors(vector<string> tids, int timestamp) {
    extend_pool(timestamp, tids.size());
    for (int i = 0; i < tids.size(); ++i) {
        auto original_tensor = id2originalTensor[tids[i]];
        auto tensor = make_shared<Tensor>(tids[i] + RECOMPUTE_SUFFIX, timestamp + i, original_tensor->free + tids.size(), original_tensor->size);
        id2tensor[tids[i]] = tensor;
        MemoryAddress best = get_best_address(tensor, infos.begin(), infos.end());
        tensor2address[tensor] = make_pair(best.first, best.first + tensor->size);
    }
}


void GreedyAllocator::allocate(string tid) {
    if (tid.find(":") == string::npos) {
        allocated_sequence.push_back(tid);
    }
}

size_t GreedyAllocator::current_size(int timestamp) {
    size_t cur_size = 0;
    for (auto t: infos) {
        if (t->alloc <= timestamp && timestamp < t->free) {
            cur_size = max(cur_size, tensor2address[t].second);
        }
    }
    return cur_size;
}

void GreedyAllocator::heuristic_alloc() {
    // The loading logic fully expanded is as follows:
    //
    // void load_logic() {
    //     // load_info
    //     if noRecompute {
    //         infos <- data/heu_info/<model>/<model>.<batch>.heu_info.txt
    //     } else {
    //         infos <- data/heu_info/<model>/<model>.<batch>.<budget>.heu_info.txt
    //     }
    //
    //     if infos != empty: return;
    //
    //     // load_info_via_exe_seq
    //     if  noRecompute {
    //         heuristic/execution/<model>/<model>.execution.txt -> data/heu_info/<model>/<model>.<batch>.heu_info.txt
    //         infos <- data/heu_info/<model>/<model>.<batch>.heu_info.txt
    //     } else {
    //         heuristic/execution/<model>/<model>.<batch>.<budget>.execution.txt -> data/heu_info/<model>/<model>.<batch>.<budget>.heu_info.txt
    //         infos <- data/heu_info/<model>/<model>.<batch>.heu_info.txt
    //     }
    //     return;
    // }
    if (infos.empty()) {
        if (!load_info()) {
            load_info_via_exe_seq();
        }
    }

    if (infos.empty()) {
        debug_print("%s: error to heuristic_alloc via empty infos\n", __FUNCTION__)
        return;
    }
    debug_print("%s: %lu infos\n", __FUNCTION__, infos.size())

    // sort the infos based on the definition of `operator<` for `GreedyAllocator::Tensor`
    // which has the following logic:
    //  1. Longest lifetime first
    //  2. Largest size first
    //  3. Tie-breaker: earlest alloc first or earlest free first
    sort(infos.begin(), infos.end());

    debug_print("infos.size = %lu\n", infos.size())
    for (int i = 0; i < infos.size(); i++) {
        debug_print("info[%d] = {id(%s), alloc(%d), free(%d), size(%zu)}\n", i, infos[i]->id.c_str(), infos[i]->alloc, infos[i]->free, infos[i]->size)
    }
    debug_print("\n")
    for (int i = 0; i < infos.size(); i++) {
//        debug_print("alloc for Tensor[%d]\n", i);
        if (i % 100 == 0) {
            debug_print("%s: try alloc [%4d/%lu]\n", __FUNCTION__, i, infos.size())
        }
        MemoryAddress best = get_best_address(infos[i], infos.begin(), infos.begin() + i);
        tensor2address[infos[i]] = make_pair(best.first, best.first + infos[i]->size);
    }
//    size_t max_address = 0;
//    for (int i = 0; i < infos.size(); i++) {
//        max_address = max(max_address, heuristic_address[i].first + profiler->tensor_size[infos[i].id]);
//        debug_print("%s\t%zu\n", infos[i].id.c_str(), heuristic_address[i].first);
//    }
//    debug_print("max-address:%zu(%lu)\n", max_address, to_string(max_address).size());
    dump_heuristic_result();
}

bool GreedyAllocator::load_info() {
    string filename = "data/heu_info/" + profiler->modelname + "/" +
                      profiler->modelname + "." + to_string(profiler->batchsize) + "." + to_string(budget_mb) + ".heu_info.txt";
    if (noRecompute) {
        filename = "data/heu_info/" + profiler->modelname + "/" +
                   profiler->modelname + "." + to_string(profiler->batchsize) + ".heu_info.txt";
    }

    ifstream ifs(filename);
    if (!ifs.is_open()) {
        debug_print("%s: error to load infos\n", __FUNCTION__)
        return false;
    }
    string i;
    int a, f, s, maxf = 0;
    while (ifs >> i >> a >> f >> s) {  // free, alloc, size
        auto t = make_shared<Tensor>(i, a, f, s);
        infos.emplace_back(t);
        id2tensor[i] = t;
        maxf = max(f, maxf);
    }
    ifs.close();
    id2originalTensor = id2tensor;
    return true;
}

template<class Iter>
GreedyAllocator::MemoryAddress GreedyAllocator::get_best_address(shared_ptr<Tensor> tensor, Iter begin, Iter end) {
    vector<MemoryAddress> madd;
    for (auto it = begin; it != end; it++) {
        if (overlap(*it, tensor)) {  // the lifetime of two tensor overlap
            auto pos = lower_bound(madd.begin(), madd.end(), tensor2address[*it]) - madd.begin();
            if (pos != 0 && mergeable(madd[pos - 1], tensor2address[*it])) {
                madd[pos - 1].first = min(madd[pos - 1].first, tensor2address[*it].first);
                madd[pos - 1].second = max(madd[pos - 1].second, tensor2address[*it].second);
                pos--;
            } else {
                madd.insert(madd.begin() + pos, tensor2address[*it]);
            }
            while (pos + 1 < madd.size() && mergeable(madd[pos], madd[pos + 1])) {
                madd[pos].first = min(madd[pos].first, madd[pos + 1].first);
                madd[pos].second = max(madd[pos].second, madd[pos + 1].second);
                madd.erase(madd.begin() + pos + 1);
            }
        }
    }
    if (madd.empty()) {
        madd.emplace_back(0, 0);
    }
    MemoryAddress best(-1, -1);
    if (madd[0].first >= tensor->size && (best.first == -1 || madd[0].first < best.second - best.first)) {
        best = MemoryAddress(0, madd[0].first);
    }
    for (int iter = 0; iter + 1 < madd.size(); iter++) {
        if (madd[iter + 1].first - madd[iter].second >= tensor->size &&
            (best.first == -1 || madd[iter + 1].first - madd[iter].second < best.second - best.first)) {
            best = MemoryAddress(madd[iter].second, madd[iter + 1].first);
        }
    }
    if (best.first == -1) {
        best = MemoryAddress(madd[madd.size() - 1].second, numeric_limits<size_t>::max());
    }
    return best;
}

void GreedyAllocator::dump_heuristic_result() {
    string filename = "heuristic/allocation/" + profiler->modelname + "/" +
                      profiler->modelname + "." + to_string(profiler->batchsize) + "." + to_string(budget_mb) + ".address.txt";
    if (noRecompute) {
        filename = "heuristic/allocation/" + profiler->modelname + "/" +
                   profiler->modelname + "." + to_string(profiler->batchsize) + ".address.txt";
    }
    for (auto iter: tensor2address) {
        max_address = max(max_address, iter.second.second);
    }
    ofstream ofs(filename, ios::out);
    if (!ofs.is_open()) {
        debug_print("%s: error to open %s\n", __FUNCTION__, filename.c_str())
        return;
    }
    debug_print("maxsize = %lu\n", max_address)
    ofs << "maxsize\t" << max_address << "\n";
    for (auto iter: tensor2address) {
        ofs << iter.first->id << "\t" << iter.second.first << "\n";
    }
    ofs.close();
}
