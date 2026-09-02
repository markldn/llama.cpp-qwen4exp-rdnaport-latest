#include "llama-moe-expert-cache.h"

#include "../ggml/src/ggml-backend-impl.h" // ggml_backend_buffer_i / ggml_backend_buffer_init: same private-header
                                             // access pattern tests/test-alloc.cpp already uses for a custom buffer.
#include "ggml-alloc.h"
#include "ggml-cuda.h" // typedefs only (ggml_backend_cuda_host_register_mapped_t) -- the actual symbols are
                        // resolved dynamically via ggml_backend_reg_get_proc_address below, never linked directly,
                        // since the CUDA/HIP backend is a separate .so loaded at runtime (ggml_backend_load_all()).

#include "llama-impl.h" // LLAMA_LOG_*

#include <cstdio>
#include <cstring>

// -- host_src's custom "GPU buffer" wrapper: a real ggml_backend_buffer_t whose
// .buft is the genuine GPU device buffer type (so the scheduler's backend/copy
// resolution treats it as already resident there -- see
// ggml_backend_sched_backend_from_buffer, which only checks buft identity) but
// whose backing memory is the pre-registered+mapped host pointer, not a fresh
// device allocation. No data ever moves through this buffer; it just lets ggml's
// bookkeeping agree with what the GPU can already dereference directly.

static void * host_view_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static enum ggml_status host_view_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor *) {
    return GGML_STATUS_SUCCESS;
}

static void host_view_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    std::memset((char *) tensor->data + offset, value, size);
}

static void host_view_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::memcpy((char *) tensor->data + offset, data, size);
}

static void host_view_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    std::memcpy(data, (const char *) tensor->data + offset, size);
}

static void host_view_buffer_clear(ggml_backend_buffer_t, uint8_t) {
    // no-op: this buffer aliases the model's own weight memory, never cleared
}

static const ggml_backend_buffer_i host_view_buffer_iface = {
    /* .free_buffer   = */ nullptr, // we don't own the memory (the weight tensor / mmap does)
    /* .get_base      = */ host_view_buffer_get_base,
    /* .init_tensor   = */ host_view_buffer_init_tensor,
    /* .memset_tensor = */ host_view_buffer_memset_tensor,
    /* .set_tensor    = */ host_view_buffer_set_tensor,
    /* .get_tensor    = */ host_view_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ host_view_buffer_clear,
    /* .reset         = */ nullptr,
};

llama_moe_expert_cache::llama_moe_expert_cache(ggml_backend_dev_t dev, int32_t cache_size) :
    m_dev(dev), m_cache_size(cache_size) {

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    m_host_register_mapped_fn   = ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_register_mapped");
    m_host_unregister_mapped_fn = ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unregister_mapped");

    if (!m_host_register_mapped_fn) {
        LLAMA_LOG_WARN("%s: backend %s does not expose mapped host-memory registration; "
                        "the MoE expert cache will refuse every register_weight() call\n",
                        __func__, ggml_backend_dev_name(dev));
    }
}

llama_moe_expert_cache::~llama_moe_expert_cache() {
    if (!m_host_unregister_mapped_fn) {
        return;
    }
    auto unregister_fn = (ggml_backend_cuda_host_unregister_mapped_t) m_host_unregister_mapped_fn;
    for (auto & [weight, layer] : m_layers) {
        if (layer.host_registered_ptr) {
            unregister_fn(layer.host_registered_ptr);
        }
    }
}

bool llama_moe_expert_cache::has(const ggml_tensor * weight) const {
    return m_layers.find(weight) != m_layers.end();
}

bool llama_moe_expert_cache::register_weight(const ggml_tensor * weight, int32_t n_expert_used_max) {
    if (!m_host_register_mapped_fn) {
        return false;
    }
    if (has(weight)) {
        return true; // already registered (e.g. re-entrant model load path)
    }
    GGML_ASSERT(n_expert_used_max <= m_cache_size);
    if (weight->data == nullptr) {
        // llama_context can be constructed speculatively (e.g. common_fit_params'
        // VRAM-fitting dry run) before the model's tensor data is actually loaded;
        // silently decline here rather than abort -- the real load's llama_context
        // construction will call register_weight again with valid data.
        return false;
    }
    GGML_ASSERT(ggml_is_contiguous(weight));

    const int32_t num_experts = (int32_t) weight->ne[2];
    const int64_t row_elems   = weight->ne[0] * weight->ne[1];
    const size_t  nbytes      = ggml_nbytes(weight);

    auto register_fn = (ggml_backend_cuda_host_register_mapped_t) m_host_register_mapped_fn;
    void * device_ptr = nullptr;
    if (!register_fn(weight->data, nbytes, &device_ptr)) {
        LLAMA_LOG_WARN("%s: failed to pin+map %s (%.1f MiB); leaving it on the static "
                        "--n-cpu-moe CPU path\n", __func__, weight->name, nbytes / 1024.0 / 1024.0);
        return false;
    }

    layer_state layer;
    layer.host_registered_ptr = weight->data;

    // GPU-resident bookkeeping + pool, own small ggml_context/buffer per layer.
    {
        ggml_init_params iparams{};
        iparams.mem_size = 16 * ggml_tensor_overhead();
        iparams.no_alloc = true;
        layer.gpu_ctx = ggml_context_ptr(ggml_init(iparams));
        ggml_context * ctx = layer.gpu_ctx.get();

        layer.pool       = ggml_new_tensor_2d(ctx, weight->type, row_elems, m_cache_size);
        layer.slot_of_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, num_experts);
        layer.id_of_slot = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, m_cache_size);
        layer.usage      = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, m_cache_size);
        layer.step       = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        // Sized to cache_size, not n_expert_used_max: apply()'s caller (build_lora_mm_id)
        // only engages the cache when ggml_nelements(ids) <= cache_size, and a call's
        // misses can never exceed its total ids count -- so cache_size is the true
        // worst case here, and must match what that gate actually checks (n_expert_used
        // alone under-covers multi-token calls, e.g. a short warmup prompt).
        layer.src_idx    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, m_cache_size);
        layer.dst_idx    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, m_cache_size);
        layer.num_copy   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

        ggml_backend_buffer_type_t gpu_buft = ggml_backend_dev_buffer_type(m_dev);
        layer.gpu_buf = ggml_backend_buffer_ptr(ggml_backend_alloc_ctx_tensors_from_buft(ctx, gpu_buft));
        if (!layer.gpu_buf) {
            LLAMA_LOG_WARN("%s: failed to allocate GPU cache buffer for %s\n", __func__, weight->name);
            if (m_host_unregister_mapped_fn) {
                ((ggml_backend_cuda_host_unregister_mapped_t) m_host_unregister_mapped_fn)(weight->data);
            }
            return false;
        }

        // cold-start LRU state
        std::vector<int32_t> neg1_experts(num_experts, -1);
        std::vector<int32_t> neg1_slots(m_cache_size, -1);
        std::vector<int64_t> zeros_slots(m_cache_size, 0);
        int64_t zero64 = 0;
        ggml_backend_tensor_set(layer.slot_of_id, neg1_experts.data(), 0, ggml_nbytes(layer.slot_of_id));
        ggml_backend_tensor_set(layer.id_of_slot, neg1_slots.data(), 0, ggml_nbytes(layer.id_of_slot));
        ggml_backend_tensor_set(layer.usage, zeros_slots.data(), 0, ggml_nbytes(layer.usage));
        ggml_backend_tensor_set(layer.step, &zero64, 0, sizeof(int64_t));
    }

    // host_src: a real tensor whose buffer's TYPE is the GPU device's own buffer
    // type (so the scheduler resolves it to that backend, no copy inserted) but
    // whose backing memory is the mapped host pointer we just registered.
    {
        ggml_init_params iparams{};
        iparams.mem_size = 4 * ggml_tensor_overhead();
        iparams.no_alloc = true;
        layer.host_ctx = ggml_context_ptr(ggml_init(iparams));

        ggml_tensor * host_src = ggml_new_tensor_2d(layer.host_ctx.get(), weight->type, row_elems, num_experts);
        ggml_format_name(host_src, "%s.moe_cache_host_src", weight->name);

        ggml_backend_buffer_type_t gpu_buft = ggml_backend_dev_buffer_type(m_dev);
        layer.host_view_buf = ggml_backend_buffer_ptr(
            ggml_backend_buffer_init(gpu_buft, host_view_buffer_iface, device_ptr, nbytes));
        ggml_backend_tensor_alloc(layer.host_view_buf.get(), host_src, device_ptr);

        layer.host_src = host_src;
    }

    m_layers.emplace(weight, std::move(layer));
    LLAMA_LOG_INFO("%s: registered %s for the MoE expert cache (%d experts, cache_size=%d, %.1f MiB pinned+mapped)\n",
                    __func__, weight->name, num_experts, m_cache_size, nbytes / 1024.0 / 1024.0);
    return true;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_moe_expert_cache::apply(
        ggml_context * ctx0, const ggml_tensor * weight, ggml_tensor * ids) const {

    auto it = m_layers.find(weight);
    GGML_ASSERT(it != m_layers.end());
    const layer_state & layer = it->second;

    // Unlike ggml_mul_mat_id/ggml_add_id (which address `ids` via explicit nb[]
    // strides), ggml_moe_lru_ensure's kernel reads it as a flat array -- so it
    // needs a genuinely contiguous tensor. `ids` (the router's selected-experts
    // output) is not guaranteed contiguous for n_tokens > 1; ggml_cont is a cheap
    // copy at this size (a few dozen int32s) and the safest fix given the kernel
    // already exists and works, vs. teaching it stride-aware addressing.
    ggml_tensor * ids_cont = ggml_cont(ctx0, ids);

    ggml_tensor * remapped = ggml_moe_lru_ensure(
        ctx0, ids_cont, layer.slot_of_id, layer.id_of_slot, layer.usage, layer.step,
        layer.src_idx, layer.dst_idx, layer.num_copy, m_cache_size);

    ggml_tensor * pool_view = ggml_moe_expert_copy(
        ctx0, layer.src_idx, layer.dst_idx, layer.num_copy, layer.host_src, layer.pool);

    // ggml's dependency tracker only sees data flow through a node's `src[]` array;
    // src_idx/dst_idx/num_copy are declared as INPUTS to both ops above (LRU_ENSURE
    // mutates them in place, which ggml's graph builder has no way to know), so
    // without this there is no edge forcing LRU_ENSURE to execute before
    // EXPERT_COPY -- they can race, and EXPERT_COPY reading not-yet-written
    // dst_idx/num_copy is exactly what produced the out-of-bounds pool writes
    // this fixes. `remapped` is LRU_ENSURE's real declared output, so adding it
    // as an (unused by the kernel, ordering-only) extra src here makes ggml's
    // topological sort schedule LRU_ENSURE strictly first.
    GGML_ASSERT(pool_view->src[5] == nullptr);
    pool_view->src[5] = remapped;

    // pool/host_src are 2D ([row_elems, cache_size]) so the copy kernel can treat each
    // expert as one flat contiguous chunk; mul_mat_id needs the original 3D per-expert
    // shape back -- a free reshape view over the same (already correct) memory.
    ggml_tensor * pool_3d = ggml_reshape_3d(ctx0, pool_view, weight->ne[0], weight->ne[1], m_cache_size);

    return { pool_3d, remapped };
}
