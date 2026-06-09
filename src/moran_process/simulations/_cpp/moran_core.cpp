// moran_core.cpp
//
// C++ implementation of the Moran process simulation, exposed to Python via
// pybind11 as the extension module `_moran_cpp`.
//
// This mirrors src/moran_process/simulations/moran_simulation_process.py but is
// designed for *statistical* equivalence, not bit-exact reproduction of NumPy's
// PCG64 stream. Given the same seed the per-trajectory results differ from the
// Python reference, but the aggregate quantities the thesis cares about
// (fixation probability rho and the fixation-time distribution) match within
// Monte Carlo error.
//
// Key design choices (see also the accompanying explanation):
//   * RNG: xoshiro256++ seeded via splitmix64.
//   * Reproducer pick: two-pool O(1) sampling. Mutants all share fitness r and
//     wild-types all share fitness 1, so picking proportional to fitness reduces
//     to choosing the mutant pool with probability M*r / F and then a uniform
//     member. This is the same per-node distribution as NumPy's cumulative-sum
//     choice, in constant time.
//   * State partition: `order` holds node indices with mutants in [0, M) and
//     wild-types in [M, n); `loc` is the inverse map. A flip is one swap.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// xoshiro256++ PRNG
// ---------------------------------------------------------------------------
namespace {

inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// splitmix64: expands a single 64-bit seed into a well-mixed stream, used to
// initialise the four xoshiro256++ state words (avoids the all-zero state).
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

class Xoshiro256pp {
public:
    // seed < 0 (e.g. -1 from Python's seed=None) => draw entropy from the OS,
    // matching NumPy's default_rng(None) behaviour.
    explicit Xoshiro256pp(int64_t seed) {
        uint64_t s;
        if (seed < 0) {
            std::random_device rd;
            s = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        } else {
            s = static_cast<uint64_t>(seed);
        }
        SplitMix64 sm(s);
        for (int i = 0; i < 4; ++i) {
            s_[i] = sm.next();
        }
    }

    inline uint64_t next() {
        const uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform double in [0, 1) with 53 bits of precision (same construction as
    // NumPy's Generator.random()).
    inline double next_double() {
        return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    }

    // Unbiased integer in [0, n) via Lemire's multiply-shift with rejection.
    inline uint64_t bounded(uint64_t n) {
        uint64_t x = next();
        __uint128_t m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(n);
        uint64_t l = static_cast<uint64_t>(m);
        if (l < n) {
            uint64_t t = (-n) % n;
            while (l < t) {
                x = next();
                m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(n);
                l = static_cast<uint64_t>(m);
            }
        }
        return static_cast<uint64_t>(m >> 64);
    }

private:
    uint64_t s_[4];
};

}  // namespace

// ---------------------------------------------------------------------------
// MoranProcessCore
// ---------------------------------------------------------------------------
class MoranProcessCore {
public:
    MoranProcessCore(int n_nodes,
                     py::array_t<int32_t, py::array::c_style | py::array::forcecast> nbrs,
                     py::array_t<int32_t, py::array::c_style | py::array::forcecast> offsets,
                     double selection_coefficient,
                     int64_t max_steps,
                     int64_t seed)
        : n_(n_nodes),
          r_(selection_coefficient),
          max_steps_(max_steps),
          rng_(seed),
          mutant_count_(0) {
        // Copy CSR arrays into owned vectors for cache locality. This happens
        // once per task (not per repeat), so the cost is negligible.
        auto nbrs_buf = nbrs.unchecked<1>();
        auto off_buf = offsets.unchecked<1>();
        nbrs_.resize(nbrs_buf.shape(0));
        for (py::ssize_t i = 0; i < nbrs_buf.shape(0); ++i) {
            nbrs_[i] = nbrs_buf(i);
        }
        offsets_.resize(off_buf.shape(0));
        for (py::ssize_t i = 0; i < off_buf.shape(0); ++i) {
            offsets_[i] = off_buf(i);
        }

        state_.assign(n_, 0);
        order_.resize(n_);
        loc_.resize(n_);
        for (int i = 0; i < n_; ++i) {
            order_[i] = i;
            loc_[i] = i;
        }
    }

    // Place n_mutants mutants at distinct, uniformly chosen nodes. Returns the
    // chosen node indices (parity with the Python reference, which the worker
    // does not actually consume).
    std::vector<int> initialize_random_mutant(int n_mutants) {
        if (n_mutants > n_) {
            throw std::invalid_argument(
                "Number of mutants exceeds number of nodes in the graph.");
        }
        // Reset to all wild-type.
        std::fill(state_.begin(), state_.end(), 0);
        for (int i = 0; i < n_; ++i) {
            order_[i] = i;
            loc_[i] = i;
        }
        mutant_count_ = 0;

        std::vector<int> chosen;
        chosen.reserve(n_mutants);
        // Repeatedly pick a uniform wild-type node from the [M, n) region and
        // flip it. Each pick is distinct => a uniform sample without replacement.
        for (int k = 0; k < n_mutants; ++k) {
            uint64_t idx = mutant_count_ +
                           rng_.bounded(static_cast<uint64_t>(n_ - mutant_count_));
            int node = order_[idx];
            add_mutant(node);
            chosen.push_back(node);
        }
        return chosen;
    }

    int mutant_count() const { return mutant_count_; }
    double selection_coeff() const { return r_; }

    py::dict run(bool track_history) {
        auto start_time = std::chrono::steady_clock::now();

        int64_t steps = 0;
        bool fixation = false;
        int initial_mutants = mutant_count_;

        std::vector<int> history;

        while (steps < max_steps_) {
            if (track_history) {
                history.push_back(mutant_count_);
            }
            if (mutant_count_ == 0) {
                break;
            }
            if (mutant_count_ == n_) {
                fixation = true;
                break;
            }
            step();
            ++steps;
        }

        auto end_time = std::chrono::steady_clock::now();
        double duration =
            std::chrono::duration<double>(end_time - start_time).count();

        py::dict result;
        result["fixation"] = fixation;
        result["steps"] = steps;
        result["initial_mutants"] = initial_mutants;
        result["selection_coeff"] = r_;
        result["duration"] = duration;
        if (track_history) {
            result["history"] = py::array_t<int>(history.size(), history.data());
        }
        return result;
    }

    // Run n_repeats independent simulations back-to-back, each preceded by a
    // fresh random mutant placement, advancing this object's single RNG stream.
    // Returns three equal-length NumPy arrays (fixation/steps/duration) so the
    // whole task crosses the Python<->C++ boundary exactly once instead of
    // 2*n_repeats times. The absorption loop is intentionally inlined here
    // (rather than shared with run()) to keep the validated, history-capable
    // run() untouched.
    py::dict run_repeats(int64_t n_repeats, int n_mutants) {
        auto fixation_arr = py::array_t<bool>(n_repeats);
        auto steps_arr = py::array_t<int64_t>(n_repeats);
        auto duration_arr = py::array_t<double>(n_repeats);
        auto fx = fixation_arr.mutable_unchecked<1>();
        auto st = steps_arr.mutable_unchecked<1>();
        auto du = duration_arr.mutable_unchecked<1>();

        for (int64_t rep = 0; rep < n_repeats; ++rep) {
            initialize_random_mutant(n_mutants);

            auto start_time = std::chrono::steady_clock::now();
            int64_t steps = 0;
            bool fixation = false;
            while (steps < max_steps_) {
                if (mutant_count_ == 0) break;
                if (mutant_count_ == n_) {
                    fixation = true;
                    break;
                }
                step();
                ++steps;
            }
            auto end_time = std::chrono::steady_clock::now();

            fx(rep) = fixation;
            st(rep) = steps;
            du(rep) =
                std::chrono::duration<double>(end_time - start_time).count();
        }

        py::dict result;
        result["fixation"] = fixation_arr;
        result["steps"] = steps_arr;
        result["duration"] = duration_arr;
        return result;
    }

private:
    // One Moran step: fitness-weighted reproducer, uniform-random neighbour
    // victim. Only called when 0 < M < n, so both pools are non-empty.
    inline void step() {
        const double F = static_cast<double>(n_ - mutant_count_) +
                         static_cast<double>(mutant_count_) * r_;
        const double u = rng_.next_double() * F;

        int reproducer;
        if (u < static_cast<double>(mutant_count_) * r_) {
            uint64_t k = rng_.bounded(static_cast<uint64_t>(mutant_count_));
            reproducer = order_[k];
        } else {
            uint64_t k =
                rng_.bounded(static_cast<uint64_t>(n_ - mutant_count_));
            reproducer = order_[mutant_count_ + k];
        }

        const int beg = offsets_[reproducer];
        const int end = offsets_[reproducer + 1];
        const int deg = end - beg;
        if (deg > 0) {
            uint64_t k = rng_.bounded(static_cast<uint64_t>(deg));
            int victim = nbrs_[beg + k];
            int rs = state_[reproducer];
            if (state_[victim] != rs) {
                if (rs == 1) {
                    add_mutant(victim);
                } else {
                    remove_mutant(victim);
                }
            }
        }
    }

    // Swap two positions in `order` and keep `loc` consistent.
    inline void swap_order(int a, int b) {
        int na = order_[a];
        int nb = order_[b];
        order_[a] = nb;
        order_[b] = na;
        loc_[nb] = a;
        loc_[na] = b;
    }

    // Flip a wild-type node to mutant: move it to the boundary slot M, grow M.
    inline void add_mutant(int v) {
        swap_order(loc_[v], mutant_count_);
        state_[v] = 1;
        ++mutant_count_;
    }

    // Flip a mutant node to wild-type: move it to the last mutant slot, shrink M.
    inline void remove_mutant(int v) {
        swap_order(loc_[v], mutant_count_ - 1);
        state_[v] = 0;
        --mutant_count_;
    }

    int n_;
    double r_;
    int64_t max_steps_;
    Xoshiro256pp rng_;
    int mutant_count_;

    std::vector<int> nbrs_;     // CSR concatenated neighbour lists
    std::vector<int> offsets_;  // CSR offsets, length n+1
    std::vector<int> state_;    // 0 = wild-type, 1 = mutant
    std::vector<int> order_;    // node indices: mutants in [0, M), wild in [M, n)
    std::vector<int> loc_;      // inverse of order_: loc_[order_[i]] == i
};

PYBIND11_MODULE(_moran_cpp, m) {
    m.doc() = "C++ Moran process core (statistical equivalent of MoranProcess)";

    py::class_<MoranProcessCore>(m, "MoranProcessCore")
        .def(py::init<int,
                      py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
                      py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
                      double, int64_t, int64_t>(),
             py::arg("n_nodes"), py::arg("nbrs"), py::arg("offsets"),
             py::arg("selection_coefficient"), py::arg("max_steps"),
             py::arg("seed"))
        .def("initialize_random_mutant",
             &MoranProcessCore::initialize_random_mutant, py::arg("n_mutants") = 1)
        .def("run", &MoranProcessCore::run, py::arg("track_history") = false)
        .def("run_repeats", &MoranProcessCore::run_repeats,
             py::arg("n_repeats"), py::arg("n_mutants") = 1)
        .def_property_readonly("mutant_count", &MoranProcessCore::mutant_count)
        .def_property_readonly("selection_coeff",
                               &MoranProcessCore::selection_coeff);
}
