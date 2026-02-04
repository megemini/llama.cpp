#include "models.h"



llm_build_ernie4_5_vl_moe::llm_build_ernie4_5_vl_moe(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);


    // todo(megemini):
    int sections[4] = {22, 22, 20, 0};


    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    GGML_ASSERT(hparams.n_moe_layer_step > 0 && "Ernie 4.5 MoE requires n_moe_layer_step > 0");
    // fprintf(stderr, "[DEBUG] Starting layer loop: n_layer=%d, n_layer_dense_lead=%d, n_moe_layer_step=%d\n",
            // n_layer, hparams.n_layer_dense_lead, hparams.n_moe_layer_step);

    for (int il = 0; il < n_layer; ++il) {
        // fprintf(stderr, "[DEBUG] ========== Layer %d start ==========\n", il);
        ggml_tensor * inpSA = inpL;
        // norm
        {
            // fprintf(stderr, "[DEBUG] Layer %d: Processing attention norm\n", il);
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);
            // fprintf(stderr, "[DEBUG] Layer %d: Completed attention norm\n", il);
        }
        // self-attention
        {
            // fprintf(stderr, "[DEBUG] Layer %d: Processing self-attention\n", il);
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);
            }
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);
            }
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // todo(megemini): GGML_ROPE_TYPE_MROPE
            if (ubatch.embd) {
                Qcur = ggml_rope_multi(
                        ctx0, Qcur, inp_pos, nullptr,
                        // n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                        n_rot, sections, GGML_ROPE_TYPE_MROPE, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_multi(
                        ctx0, Kcur, inp_pos, nullptr,
                        // n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                        n_rot, sections, GGML_ROPE_TYPE_MROPE, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );
            } else {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                    ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                    ext_factor, attn_factor, beta_fast, beta_slow);


            }


            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "attn_out", il);
            // fprintf(stderr, "[DEBUG] Layer %d: Completed self-attention\n", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            // fprintf(stderr, "[DEBUG] Layer %d: Last layer detected, applying inp_out_ids\n", il);
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);
        // fprintf(stderr, "[DEBUG] Layer %d: Prepared FFN input\n", il);

        // feed-forward network
        bool is_moe_layer =
            static_cast<uint32_t>(il) >= hparams.n_layer_dense_lead && (il + 1) % hparams.n_moe_layer_step == 0;

        // fprintf(stderr, "[DEBUG] Layer %d: is_moe_layer=%d\n", il, is_moe_layer);

        if (!is_moe_layer) {
            // fprintf(stderr, "[DEBUG] Layer %d: Processing standard FFN branch (dense layer)\n", il);
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
            // fprintf(stderr, "[DEBUG] Layer %d: Completed ffn_norm\n", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
            // fprintf(stderr, "[DEBUG] Layer %d: Completed standard FFN\n", il);
        } else {
            // MoE branch
            // fprintf(stderr, "[DEBUG] Layer %d: Processing MoE branch\n", il);
            // fprintf(stderr, "[DEBUG] Layer %d: is_moe_layer=%d, n_layer_dense_lead=%d, n_moe_layer_step=%d\n",
                    // il, is_moe_layer, hparams.n_layer_dense_lead, hparams.n_moe_layer_step);

            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
            // fprintf(stderr, "[DEBUG] Layer %d: Completed ffn_norm\n", il);

            // todo(megemini):
            ggml_tensor * moe_out = nullptr;




            // todo(megemini):
            if (ubatch.embd) {
                // fprintf(stderr, "[DEBUG] Layer %d: Using v_ffn (vision) path, n_expert=%d, n_expert_used=%d\n",
                        // il, n_expert, n_expert_used);
                moe_out = build_moe_ffn(cur,
                                            model.layers[il].ffn_gate_inp,
                                            model.layers[il].ffn_up_exps,
                                            model.layers[il].ffn_gate_exps,
                                            model.layers[il].ffn_down_exps,
                                            model.layers[il].ffn_exp_probs_b,
                                            n_expert, n_expert_used,
                                            LLM_FFN_SILU, true,
                                            false, 0.0,
                                            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                                            il);

                cb(moe_out, "ffn_moe_out", il);
                // fprintf(stderr, "[DEBUG] Layer %d: Completed v_ffn MoE\n", il);

            } else {
                // fprintf(stderr, "[DEBUG] Layer %d: Using standard ffn path, n_expert=%d, n_expert_used=%d\n",
                        // il, n_expert, n_expert_used);
                moe_out = build_moe_ffn(cur,
                                            model.layers[il].ffn_gate_inp,
                                            model.layers[il].ffn_up_exps,
                                            model.layers[il].ffn_gate_exps,
                                            model.layers[il].ffn_down_exps,
                                            model.layers[il].ffn_exp_probs_b,
                                            n_expert, n_expert_used,
                                            LLM_FFN_SILU, true,
                                            false, 0.0,
                                            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                                            il);
                cb(moe_out, "ffn_moe_out", il);
                // fprintf(stderr, "[DEBUG] Layer %d: Completed standard MoE\n", il);
            }


            // Shared expert (if present)
            // fprintf(stderr, "[DEBUG] Layer %d: n_ff_shexp=%d (shared expert flag)\n", il, hparams.n_ff_shexp);
            if (hparams.n_ff_shexp > 0) {
                // fprintf(stderr, "[DEBUG] Layer %d: Processing shared expert\n", il);
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                // fprintf(stderr, "[DEBUG] Layer %d: Added shared expert output to MoE output\n", il);
            } else {
                // fprintf(stderr, "[DEBUG] Layer %d: No shared expert, using MoE output directly\n", il);
                cur = moe_out;
            }
            cb(cur, "ffn_out", il);
            // fprintf(stderr, "[DEBUG] Layer %d: Completed MoE branch\n", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_residual", il);
        // fprintf(stderr, "[DEBUG] Layer %d: Added FFN residual\n", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        // fprintf(stderr, "[DEBUG] Layer %d: Completed cvec transformation\n", il);

        // input for next layer
        inpL = cur;
        // fprintf(stderr, "[DEBUG] ========== Layer %d end ==========\n", il);
    }
    // fprintf(stderr, "[DEBUG] Layer loop completed\n");

    cur = inpL;
    // fprintf(stderr, "[DEBUG] Starting output processing\n");

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    // fprintf(stderr, "[DEBUG] Completed output norm\n");

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    // fprintf(stderr, "[DEBUG] Building lm_head output\n");
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
    // fprintf(stderr, "[DEBUG] Completed output projection\n");

    // fprintf(stderr, "[DEBUG] Building forward graph expansion\n");
    ggml_build_forward_expand(gf, cur);
    // fprintf(stderr, "[DEBUG] Model build complete\n");
}
