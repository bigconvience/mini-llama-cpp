#include "generate.h"

/**
 * Greedy search: select token with maximum log probability
 */
u32 greedy_sample(Tensor* logits)
{
    u32 max_idx = 0;
    f32 max_val = logits->data[0];
    for (u32 i = 1; i < logits->size; i++)
    {
        if (logits->data[i] > max_val)
        {
            max_val = logits->data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/**
 * Minimal demo autoregressive generation loop
 * All weights are zero initialized for demo, no real GGUF loaded
 */
u32 generate_autoregressive(LLaMAModel* model, KVCache* cache,
    u32* input_tokens, u32 in_token_count,
    u32* out_tokens, u32 max_gen_tokens)
{
    // Copy input prompt tokens to output buffer first
    u32 total_tokens = in_token_count;
    memcpy(out_tokens, input_tokens, in_token_count * sizeof(u32));

    // Temporary tensor for single token embedding forward pass
    u32 vec_shape[] = {1, model->cfg.dim};
    Tensor* hidden = tensor_create(2, vec_shape);
    Tensor* q = tensor_create(2, vec_shape);
    Tensor* k = tensor_create(2, vec_shape);
    Tensor* v = tensor_create(2, vec_shape);
    Tensor* attn_out = tensor_create(2, vec_shape);

    // Logit output tensor: [1, vocab_size]
    u32 logit_shape[] = {1, model->cfg.vocab_size};
    Tensor* logits = tensor_create(2, logit_shape);

    // Generate tokens one by one
    for (u32 step = 0; step < max_gen_tokens; step++)
    {
        u32 current_pos = total_tokens - 1;
        // Demo: embedding lookup returns all zero vector
        memset(hidden->data, 0, hidden->size * sizeof(f32));

        // RMSNorm (now attn_norm tensor is valid, no null ptr)
        rms_norm(q->data, hidden->data, model->layers[0].attn_norm->data, model->cfg.dim);
        memcpy(k->data, q->data, model->cfg.dim * sizeof(f32));
        memcpy(v->data, q->data, model->cfg.dim * sizeof(f32));

        // RoPE positional encoding
        rope(q->data, k->data, current_pos, model->cfg.dim, model->cfg.dim / model->cfg.n_heads);

        // Causal multi-head attention with KV cache
        causal_mha(q, k, v, cache, attn_out, current_pos, model->cfg.n_heads);

        // Demo: logits all zero, greedy always picks token 0
        memset(logits->data, 0, logits->size * sizeof(f32));
        u32 next_tok = greedy_sample(logits);
        out_tokens[total_tokens++] = next_tok;

        if (next_tok == TOKEN_EOS)
            break;
    }

    tensor_free(hidden);
    tensor_free(q);
    tensor_free(k);
    tensor_free(v);
    tensor_free(attn_out);
    tensor_free(logits);
    return total_tokens;
}