#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <regex>
#include "DemoUnit.hpp"

#include <memory>
#include <limits>
#include <iterator>

using namespace std;


#define debug_print(format, ...) {printf(format, ##__VA_ARGS__);fflush(stdout);}
// #define debug_print(format, ...)


class VanillaAllocator;

class Profiler;

class Recomputer;

class GreedyAllocator;

class VanillaAllocator {
public:
    void alloc(string tid, size_t size = 0) {
        if (size) {
            cur += size;
            tsz[tid] = size;
            peak = max(peak, cur);
        }
    }

    void free(string tid) {
        cur -= tsz[tid];
    }

    size_t total_size() {
        return peak;
    }

    size_t current_size() {
        return cur;
    }

    size_t cur = 0, peak = 0;
    map<string, size_t> tsz;
};

class Profiler {
public:
    struct OpInfo {
        string opid;
        vector<string> inputs, outputs, release, temporary;

        OpInfo(string op = "") {
            opid = op;
        }

        friend ostream &operator<<(ostream &out, OpInfo &info);
    };

    Profiler(string mn, int bs);

    vector<OpInfo> io_info;
    map<string, size_t> tensor_size;
    map<string, double> cost_info;
    vector<vector<pair<string, string>>> resize_info;
    map<string, string> redundent_parent;
    string modelname;
    int batchsize, fp_thres = -1, num_layers = 0;

    void load_infomation(string basename);

    void load_io_info(string filename);

    void load_resize_info(string filename);

    void load_cost_info(string filename);

    void load_tensor_size(string filename);

    void load_redundent_parent(string filename);

    void set_thres_layers();

    void init_from_scratch();

    void profile_from_scratch(string filename);

    void resize_from_scratch(string filename);

    void cost_from_scratch(string filename);

    void add_info(string ln, vector<string> &vec);

    void dump_information();

    void dump_io_info(string filename);

    void dump_resize_info(string filename);

    void dump_cost_info(string filename);

    void dump_tensor_size(string filename);

    void dump_redundent_parent(string filename);

    void dump_original_execution_info(string filename);
};

class Recomputer {
public:
    Recomputer(shared_ptr<Profiler> profiler_ptr, shared_ptr<GreedyAllocator> allocator_ptr, size_t budget, double thres = 1.0);

    void ondemand_recompute_via_metric();

    void memory_calibrated_progressive_recomputation();

    void dump_exe_seq();

    shared_ptr<Profiler> profiler = nullptr;
    shared_ptr<GreedyAllocator> grd_allocator = nullptr;
    size_t budget_b = 0, budget_mb = 0;
    double threshold = 1.0;
    VanillaAllocator vnl_allocator;
    set<string> feature_map, allocated_tensor;
    map<string, double> metric;
    map<string, set<string>> table_in, table_out;
    vector<pair<string, string>> exe_seq;
    string current_progress;
    map<string, string> release_point;
    map<string, map<int, size_t>> computing_budget;
    int calibrated_timestamp = 0;

    set<string> get_compute_source(string ith);

    void adjust_exe_seq();

    void get_feature_map();

    void update_metric(string ith);

    void ondemand_compute(string ith, bool recompute=false, set<string> skip_pre_rel = set<string>());

    void calibrated_compute(string ith, bool recompute=false);
};

class GreedyAllocator {
public:
    struct Tensor;
    typedef pair<size_t, size_t> MemoryAddress; // [start, end)

    GreedyAllocator(shared_ptr<Profiler> profiler_ptr, size_t bgt_mb, bool norecomp = false);

    void heuristic_alloc();

    template<class Iter>
    MemoryAddress get_best_address(shared_ptr<Tensor> tensor, Iter begin, Iter end);

    bool load_info();

    bool load_info_via_exe_seq();

    void dump_heuristic_result();

    size_t current_size(int timestamp);

    bool overlap(shared_ptr<Tensor> t1, shared_ptr<Tensor> t2);

    bool overlap(MemoryAddress m1, MemoryAddress m2);

    bool mergeable(MemoryAddress m1, MemoryAddress m2);

    void build_topology();

    void remove_tensor(string tid, int timestamp);

    void extend_pool(int timestamp, int length);

    void insert_tensors(vector<string> tids, int timestamp);

    void allocate(string tid);

    shared_ptr<Profiler> profiler = nullptr;
    size_t budget_mb;
    map<shared_ptr<Tensor>, MemoryAddress> tensor2address;
    vector<shared_ptr<Tensor>> infos;
    map<shared_ptr<Tensor>, vector<shared_ptr<Tensor>>> up_tensors, down_tensors;
    map<string, shared_ptr<Tensor>> id2tensor, id2originalTensor;
    bool noRecompute = false;
    size_t max_address = 0;
    vector<string> allocated_sequence;
};

class GeneratePlan : public DemoUnit {
public:
    virtual int run(int argc, const char *argv[]) override {
        if (argc < 4) {
            std::cout << "./runTrainDemo GeneratePlan MODEL BATCH BUDGET [NORECOMPUTE_FLAG=false]\n";
            return 0;
        }
        // argv[0] is GeneratePlan here. While in cli, argv[0] is ./runTrainDemo.out
        std::string modelname = argv[1];
        int batchsize = atoi(argv[2]);
        int mem_bgt = atoi(argv[3]);
        bool noRecompute = argc == 5;
        if (mem_bgt == 6) mem_bgt = 3800;
        if (mem_bgt == 8) mem_bgt = 5500;
        if (mem_bgt != 3800 && mem_bgt != 5500) {
            cout << "the budget used to generate must be 6GB (3800MB) or 8GB (5500MB)\n";
            return 0;
        }
        debug_print("model=%s, batch=%d, budget=%d\n", modelname.c_str(), batchsize, mem_bgt)
        shared_ptr<Profiler> profiler = make_shared<Profiler>(modelname, batchsize);
        debug_print("finish load info with io_info.size = %lu\n", profiler->io_info.size())
        shared_ptr<GreedyAllocator> grd_allocator = make_shared<GreedyAllocator>(profiler, mem_bgt, noRecompute);

        // NOTE: added by us
        if (!noRecompute) {
            grd_allocator->heuristic_alloc();
        }

        shared_ptr<Recomputer> recomputer = make_shared<Recomputer>(profiler, grd_allocator, mem_bgt);
        recomputer->memory_calibrated_progressive_recomputation();
    }
};

DemoUnitSetRegister(GeneratePlan, "GeneratePlan");
