#include "models.h"

ggml_cgraph * clip_graph_ernie45vlmoe::build() {
    // ERNIE-4.5-VL-MoE Vision + Resampler:
    // 1. ViT encoder with 2D position embeddings and M-RoPE support
    // 2. Resampler with spatial conv (2x2 grouping) + optional temporal + MLP + RMS norm

    const int  batch_size         = 1;
    const int  n_pos              = n_patches;
    const int  spatial_conv_size  = hparams.spatial_conv_size;   // 2
    const int  temporal_conv_size = hparams.temporal_conv_size;  // 2
    const bool use_temporal       = hparams.use_temporal_conv;

    fprintf(stderr, "%s: building ERNIE-4.5-VL-MoE vision encoder graph\n", __func__);
    fprintf(stderr, "%s: n_patches=%d (n_patches_x=%d, n_patches_y=%d), n_layer=%d, n_head=%d, d_head=%d\n", __func__,
            n_patches, n_patches_x, n_patches_y, n_layer, n_head, d_head);

    // GGML_ASSERT(n_patches_x == n_patches_y && "only square images supported");
    GGML_ASSERT(spatial_conv_size == 2 && "ERNIE-4.5-VL-MoE requires spatial_conv_size=2");


    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);



    // Build vision encoder with patch embedding
    // Note: patch_embeddings_0 is reshaped to 4D during export for conv2d compatibility
    ggml_tensor * inp = build_inp();


    // ggml_tensor * inp_raw = build_inp_raw();

    // fprintf(stderr, "%s: inp_raw shape: [%lld, %lld, %lld, %lld]\n", __func__, inp_raw->ne[0],
    //         inp_raw->ne[1],inp_raw->ne[2],inp_raw->ne[3]);

    // ggml_tensor * inp = ggml_reshape_2d(ctx0, inp_raw, n_patches, n_embd);


    // fprintf(stderr, "%s: 1 inp shape: [%lld, %lld, %lld, %lld]\n", __func__, inp->ne[0],
    //         inp->ne[1],inp->ne[2],inp->ne[3]);

    // inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    // fprintf(stderr, "%s: 2 inp shape: [%lld, %lld, %lld, %lld]\n", __func__, inp->ne[0],
    //         inp->ne[1],inp->ne[2],inp->ne[3]);

    // inp = ggml_mul_mat(ctx0, inp, model.patch_embeddings_0);

    // fprintf(stderr, "%s: 3 inp shape: [%lld, %lld, %lld, %lld]\n", __func__, inp->ne[0],
            // inp->ne[1],inp->ne[2],inp->ne[3]);


    // if (model.patch_bias) {
    //     inp = ggml_add(ctx0, inp, model.patch_bias);
    //     cb(inp, "patch_bias", -1);
    // }


    // ERNIE-4.5-VL uses RoPE (Rotary Position Embedding), not learned position embeddings
    // Position encoding is applied within attention layers via RoPE
    // So we don't need to add position embeddings here

    ggml_tensor * inpL = inp;

    // Pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
        cb(inpL, "pre_ln", -1);
    }

    // Loop over encoder layers
    for (int il = 0; il < n_layer; il++) {
        const auto &  layer = model.layers[il];
        ggml_tensor * cur   = inpL;

        fprintf(stderr, "%s: processing encoder layer %d/%d\n", __func__, il + 1, n_layer);

        // Layernorm 1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ln1", il);

        // Self-attention
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
            if (layer.q_b) {
                Qcur = ggml_add(ctx0, Qcur, layer.q_b);
            }

            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
            if (layer.k_b) {
                Kcur = ggml_add(ctx0, Kcur, layer.k_b);
            }

            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
            if (layer.v_b) {
                Vcur = ggml_add(ctx0, Vcur, layer.v_b);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);


            // apply M-RoPE
            Qcur = ggml_rope_multi(
                ctx0, Qcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = ggml_rope_multi(
                ctx0, Kcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);



            cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // Residual
        cur  = ggml_add(ctx0, cur, inpL);
        inpL = cur;
        cb(cur, "ffn_inp", il);

        // Layernorm 2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // FFN
        cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w,
                        layer.ff_down_b, hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // Residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // Post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
    }

    ggml_tensor * embeddings = inpL;
    cb(embeddings, "vision_output", -1);

    fprintf(stderr, "%s: vision encoder completed, starting resampler projection\n", __func__);
    fprintf(stderr, "%s: vision output shape: [%lld, %lld]\n", __func__, embeddings->ne[0], embeddings->ne[1]);

    // -------------------------------------------
    // Resampler projection
    // -------------------------------------------
    // Input shape: [n_embd, n_patches] = [1280, 1600]
    // We need to group 2x2 patches: 40x40 patches -> 20x20 groups
    // Output shape: [n_embd*4, n_groups] = [5120, 400]

    const int n_groups_x = n_patches_x / spatial_conv_size;  // 40/2 = 20
    const int n_groups_y = n_patches_y / spatial_conv_size;  // 40/2 = 20
    const int n_groups   = n_groups_x * n_groups_y;          // 400

    fprintf(stderr, "%s: spatial grouping: %dx%d patches -> %dx%d groups (n_groups=%d)\n", __func__, n_patches_x,
            n_patches_y, n_groups_x, n_groups_y, n_groups);

    // Use patch_merge_permute to group 2x2 patches
    // Note: build_patch_merge_permute expects 2D input [n_embd, n_patches]
    embeddings = build_patch_merge_permute(embeddings, spatial_conv_size);

    // embeddings is now [n_embd*4, n_groups] = [5120, 400]
    cb(embeddings, "spatial_reshape", -1);
    fprintf(stderr, "%s: after spatial merge: shape [%lld, %lld]\n", __func__, embeddings->ne[0], embeddings->ne[1]);

    // Spatial linear path: Linear -> GELU -> Linear -> LayerNorm
    ggml_tensor * spatial_out = embeddings;

    fprintf(stderr, "%s: starting spatial linear transformations\n", __func__);

    // First linear: [5120, 5120] x [5120, n_groups] -> [5120, n_groups]
    spatial_out = ggml_mul_mat(ctx0, model.mm_spatial_0_w, spatial_out);
    spatial_out = ggml_add(ctx0, spatial_out, model.mm_spatial_0_b);
    cb(spatial_out, "spatial_linear_0", -1);

    // GELU
    spatial_out = ggml_gelu(ctx0, spatial_out);
    cb(spatial_out, "spatial_gelu", -1);

    // Second linear
    spatial_out = ggml_mul_mat(ctx0, model.mm_spatial_2_w, spatial_out);
    spatial_out = ggml_add(ctx0, spatial_out, model.mm_spatial_2_b);
    cb(spatial_out, "spatial_linear_2", -1);

    // LayerNorm
    spatial_out = build_norm(spatial_out, model.mm_spatial_norm_w, model.mm_spatial_norm_b, NORM_TYPE_NORMAL, eps, -1);
    cb(spatial_out, "spatial_norm", -1);

    fprintf(stderr, "%s: spatial processing completed, shape [%lld, %lld]\n", __func__, spatial_out->ne[0],
            spatial_out->ne[1]);

    ggml_tensor * resampler_out = spatial_out;

    // Debug: check shapes before MLP
    GGML_ASSERT(resampler_out->ne[0] == n_embd * spatial_conv_size * spatial_conv_size);  // should be 5120
    GGML_ASSERT(resampler_out->ne[1] == n_groups);                                        // should be 400

    // Temporal path (skip for single images)
    if (use_temporal && model.mm_temp_0_w) {
        // For video: concatenate adjacent temporal frames
        // Since we're processing single images, this path is unused.
        // Implementation would require grid_thw and complex indexing.
        fprintf(stderr, "%s: temporal convolution enabled but not implemented for single images\n", __func__);
    }

    fprintf(stderr, "%s: starting final MLP and normalization\n", __func__);
    fprintf(stderr, "%s: MLP input (resampler_out) shape: [%lld, %lld]\n", __func__, resampler_out->ne[0],
            resampler_out->ne[1]);
    fprintf(stderr, "%s: mm_mlp_w shape: [%lld, %lld]\n", __func__, model.mm_mlp_w->ne[0], model.mm_mlp_w->ne[1]);

    // Final MLP: Linear + RMSNorm
    ggml_tensor * mm_mlp_w = ggml_transpose(ctx0, model.mm_mlp_w);
    mm_mlp_w = ggml_cont(ctx0, mm_mlp_w);


    resampler_out = ggml_mul_mat(ctx0, mm_mlp_w, resampler_out);



    resampler_out = ggml_add(ctx0, resampler_out, model.mm_mlp_b);
    cb(resampler_out, "mlp", -1);

    fprintf(stderr, "%s: after MLP: shape [%lld, %lld]\n", __func__, resampler_out->ne[0], resampler_out->ne[1]);

    // RMS norm (final output normalization)
    resampler_out = build_norm(resampler_out, model.mm_after_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);
    cb(resampler_out, "after_norm", -1);

    fprintf(stderr, "%s: final output shape [%lld, %lld]\n", __func__, resampler_out->ne[0], resampler_out->ne[1]);
    fprintf(stderr, "%s: ERNIE-4.5-VL-MoE graph construction completed\n", __func__);


    // resampler_out = ggml_concat(ctx0, resampler_out, resampler_out, 0);
    // cb(resampler_out, "after_repeat", -1);

    // fprintf(stderr, "%s: after repeat: shape [%lld, %lld]\n", __func__, resampler_out->ne[0], resampler_out->ne[1]);


    // Build the graph
    ggml_build_forward_expand(gf, resampler_out);

    return gf;
}
