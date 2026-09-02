#pragma once

// Device-side GPU-resident LRU cache for MoE expert weights: pages individual
// experts into a small persistent GPU pool per call, fetched over PCIe from
// the model's already-loaded (mmap'd) host weight tensor, instead of a static
// whole-layer CPU/GPU split (--n-cpu-moe). All cache bookkeeping (hit/miss,
// eviction, slot assignment) runs as GPU kernels with no host synchronization
// per call -- see ggml_moe_lru_ensure/ggml_moe_expert_copy in ggml.h and
// ~/.claude/plans/indexed-zooming-dream.md for the full design and the reason
// this exists (a host-orchestrated first attempt regressed decode throughput).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

// One persistent LRU pool + bookkeeping set per cached MoE weight tensor
// (e.g. one per layer's ffn_gate_up_exps, one per layer's ffn_down_exps).
struct llama_moe_expert_cache {
    llama_moe_expert_cache(ggml_backend_dev_t dev, int32_t cache_size);
    ~llama_moe_expert_cache();

    llama_moe_expert_cache(const llama_moe_expert_cache &) = delete;
    llama_moe_expert_cache & operator=(const llama_moe_expert_cache &) = delete;

    // Registers a CPU-resident MoE weight tensor (already fully loaded --
    // ->data valid and stable, e.g. mmap'd by --n-cpu-moe) for caching: pins +
    // GPU-maps its host memory (no data copy or move -- the tensor's ->data
    // pointer and buffer are untouched) and allocates a GPU pool + LRU
    // bookkeeping sized for `cache_size` resident experts. `n_expert_used_max`
    // bounds how many distinct experts a single call can need (n_expert_used,
    // or that times n_tokens for a batched call) -- must be <= cache_size.
    // Returns false (logs and leaves the tensor unregistered) if pinning
    // fails or the backend doesn't support the mapped-host-memory API.
    bool register_weight(const ggml_tensor * weight, int32_t n_expert_used_max);

    // Whether `weight` was successfully registered (i.e. the graph builder
    // should route it through apply() instead of using it directly).
    bool has(const ggml_tensor * weight) const;

    // Builds the (fixed-topology) op nodes for one call in the per-graph-build
    // context `ctx0`: returns {pool_weight_view, remapped_ids} to feed into
    // ggml_mul_mat_id instead of {weight, ids}. `weight` must have been
    // registered. `ids` is the router's selected-experts tensor (I32).
    std::pair<ggml_tensor *, ggml_tensor *> apply(
            ggml_context * ctx0, const ggml_tensor * weight, ggml_tensor * ids) const;

    int32_t cache_size() const { return m_cache_size; }

private:
    struct layer_state {
        ggml_tensor * pool       = nullptr; // [row_elems, cache_size], GPU
        ggml_tensor * host_src   = nullptr; // [row_elems, num_experts], data = mapped pointer into weight's own memory
        ggml_tensor * slot_of_id = nullptr; // [num_experts], GPU I32
        ggml_tensor * id_of_slot = nullptr; // [cache_size], GPU I32
        ggml_tensor * usage      = nullptr; // [cache_size], GPU I64
        ggml_tensor * step       = nullptr; // [1], GPU I64
        ggml_tensor * src_idx    = nullptr; // [n_expert_used_max], GPU I32 scratch
        ggml_tensor * dst_idx    = nullptr; // [n_expert_used_max], GPU I32 scratch
        ggml_tensor * num_copy   = nullptr; // [1], GPU I64 scratch

        void * host_registered_ptr = nullptr; // weight->data, for unregistration on teardown

        // one ggml_context/buffer pair per registered weight: simplest correct
        // lifetime (own everything above, freed together), cache_size here is
        // always small (tens of layers at most) so the extra buffer objects
        // this costs vs. one shared batched allocation are not worth optimizing.
        ggml_context_ptr        gpu_ctx;  // owns pool/slot_of_id/id_of_slot/usage/step/src_idx/dst_idx/num_copy
        ggml_backend_buffer_ptr gpu_buf;  // backs gpu_ctx's tensors
        ggml_context_ptr        host_ctx; // owns host_src (the tensor object only -- its data aliases host_registered_ptr)
        // wraps host_registered_ptr under the GPU device's buffer TYPE (see .cpp)
        // so the scheduler treats host_src as already resident on that device --
        // no cross-backend copy gets inserted for it.
        ggml_backend_buffer_ptr host_view_buf;
    };

    ggml_backend_dev_t m_dev;
    int32_t m_cache_size;

    void * m_host_register_mapped_fn   = nullptr; // ggml_backend_cuda_host_register_mapped_t
    void * m_host_unregister_mapped_fn = nullptr; // ggml_backend_cuda_host_unregister_mapped_t

    std::unordered_map<const ggml_tensor *, layer_state> m_layers;
};
