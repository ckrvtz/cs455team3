// test_layout.cpp: layout.h (ROADMAP P0.4, §2.2, §4.3, §6.2).
// DoD: parameter counts equal §2.2 for all presets tied/untied; offsets are
// 64-byte aligned; tensors do not overlap.
#include "test.h"

#include "tslm/common.h"
#include "tslm/config.h"
#include "tslm/layout.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace tslm;

static ModelConfig preset(const char* path, int tie) {
    Config c;
    std::string err;
    CHECK(config_load_file(path, &c, &err), "%s", err.c_str());
    c.model.tie_embeddings = tie;
    return c.model;
}

// §2.2: V·C + T·C + L·(12C² + 13C) + 2C [+ V·C untied].
static size_t formula_params(const ModelConfig& m) {
    const size_t V = size_t(m.vocab_size), C = size_t(m.n_embd), T = size_t(m.max_seq_len);
    const size_t L = size_t(m.n_layer);
    size_t n = V * C + T * C + L * (12 * C * C + 13 * C) + 2 * C;
    if (!m.tie_embeddings) n += V * C;
    return n;
}

// Structural invariants shared by both layouts: padded/aligned offsets,
// numel = product(shape), no overlaps, minimal padding, unique names.
static void check_structure(const std::vector<TensorDesc>& ts, size_t buffer_len, const char* what) {
    ASSERT_TRUE(!ts.empty());
    std::vector<size_t> order(ts.size());
    for (size_t i = 0; i < ts.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return ts[a].offset < ts[b].offset; });

    size_t prev_end = 0;
    for (size_t k = 0; k < order.size(); ++k) {
        const TensorDesc& t = ts[order[k]];
        size_t prod = 1;
        for (int d = 0; d < t.ndim; ++d) prod *= size_t(t.shape[d]);
        if (prod != t.numel || t.numel == 0) {
            test::report(__FILE__, __LINE__, std::string(what) + ": bad numel for " + t.name);
        }
        if (t.offset % LAYOUT_PAD_ELEMS != 0 || (t.offset * sizeof(real_t)) % 64 != 0) {
            test::report(__FILE__, __LINE__, std::string(what) + ": misaligned " + t.name);
        }
        if (t.offset < prev_end) {
            test::report(__FILE__, __LINE__, std::string(what) + ": overlap at " + t.name);
        }
        if (t.offset - prev_end >= LAYOUT_PAD_ELEMS) {
            test::report(__FILE__, __LINE__, std::string(what) + ": excess gap before " + t.name);
        }
        prev_end = t.offset + t.numel;
    }
    EXPECT_TRUE(prev_end <= buffer_len);
    EXPECT_TRUE(buffer_len - prev_end < LAYOUT_PAD_ELEMS);
    EXPECT_EQ(buffer_len % LAYOUT_PAD_ELEMS, size_t{0});

    std::vector<std::string> names;
    for (const TensorDesc& t : ts) names.push_back(t.name);
    std::sort(names.begin(), names.end());
    EXPECT_TRUE(std::adjacent_find(names.begin(), names.end()) == names.end());
}

TEST(param_counts_match_roadmap) {
    struct Case { const char* path; int tie; size_t want; };
    const Case cases[] = {
        {"configs/base.cfg", 1, 5785088},  {"configs/base.cfg", 0, 8345088},
        {"configs/small.cfg", 1, 1709568}, {"configs/small.cfg", 0, 2989568},
        // smoke is not tabulated in §2.2; hand-computed from the formula:
        // 10000*32 + 32*32 + 1*(12*32^2 + 13*32) + 2*32 = 333,792 (+320,000 untied).
        {"configs/smoke.cfg", 1, 333792},  {"configs/smoke.cfg", 0, 653792},
    };
    for (const Case& c : cases) {
        const ModelConfig m = preset(c.path, c.tie);
        const ParamLayout L = make_param_layout(m);
        std::printf("    %-18s tied=%d  num_params=%zu  padded=%zu\n", c.path, c.tie, L.num_params,
                    L.num_params_padded);
        EXPECT_EQ(L.num_params, c.want);
        EXPECT_EQ(L.num_params, formula_params(m));
        size_t sum = 0;
        for (const TensorDesc& t : L.tensors) sum += t.numel;
        EXPECT_EQ(sum, L.num_params);
    }
}

TEST(param_structure_all_presets) {
    const char* paths[] = {"configs/smoke.cfg", "configs/small.cfg", "configs/base.cfg"};
    for (const char* p : paths) {
        for (int tie = 0; tie <= 1; ++tie) {
            const ParamLayout L = make_param_layout(preset(p, tie));
            check_structure(L.tensors, L.num_params_padded, p);
        }
    }
    // Gradcheck model from ROADMAP §10 (tiny, odd sizes).
    ModelConfig g;
    g.vocab_size = 32;
    g.max_seq_len = 8;
    g.n_layer = 2;
    g.n_embd = 16;
    g.n_head = 2;
    for (int tie = 0; tie <= 1; ++tie) {
        g.tie_embeddings = tie;
        const ParamLayout L = make_param_layout(g);
        check_structure(L.tensors, L.num_params_padded, "gradcheck");
        EXPECT_EQ(L.num_params, formula_params(g));
    }
    // Odd sizes: every preset is a multiple of 16, so exercise real padding here.
    ModelConfig o;
    o.vocab_size = 37;
    o.max_seq_len = 5;
    o.n_layer = 3;
    o.n_embd = 12;
    o.n_head = 3;
    for (int tie = 0; tie <= 1; ++tie) {
        o.tie_embeddings = tie;
        const ParamLayout L = make_param_layout(o);
        check_structure(L.tensors, L.num_params_padded, "odd params");
        EXPECT_EQ(L.num_params, formula_params(o));
        EXPECT_TRUE(L.num_params_padded > L.num_params);
        const ActLayout A = make_act_layout(o, 3, 5);
        check_structure(A.tensors, A.num_elems, "odd acts");
    }
}

TEST(param_names_shapes_decay) {
    const ModelConfig m = preset("configs/small.cfg", 1);
    const int C = m.n_embd, V = m.vocab_size;
    const ParamLayout L = make_param_layout(m);

    ASSERT_EQ(L.tensors.size(), size_t(2 + 12 * m.n_layer + 2));
    EXPECT_TRUE(L.tied);
    EXPECT_EQ(L.head_w, L.wte);
    EXPECT_EQ(L.find("head_w"), -1);
    EXPECT_EQ(L[L.wte].name, std::string("wte"));
    EXPECT_EQ(L[L.wte].shape[0], V);
    EXPECT_EQ(L[L.wte].shape[1], C);
    EXPECT_EQ(L[L.wpe].shape[0], m.max_seq_len);
    EXPECT_EQ(L.wte, 0);  // fixed order: wte first

    ASSERT_EQ(L.layers.size(), size_t(m.n_layer));
    for (int l = 0; l < m.n_layer; ++l) {
        const LayerParamIdx& x = L.layers[size_t(l)];
        const std::string p = "h" + std::to_string(l) + ".";
        EXPECT_EQ(L[x.qkv_w].name, p + "qkv_w");
        EXPECT_EQ(L[x.qkv_w].shape[0], 3 * C);  // [OC, C]
        EXPECT_EQ(L[x.qkv_w].shape[1], C);
        EXPECT_EQ(L[x.fc_w].shape[0], 4 * C);
        EXPECT_EQ(L[x.fcproj_w].shape[0], C);
        EXPECT_EQ(L[x.fcproj_w].shape[1], 4 * C);
        EXPECT_EQ(L[x.fcproj_b].numel, size_t(C));
        EXPECT_EQ(L.find(p + "ln2_b"), x.ln2_b);
    }
    EXPECT_EQ(L[L.lnf_g].name, std::string("lnf_g"));

    // decay exactly on wte, qkv_w, proj_w, fc_w, fcproj_w (§2.3).
    int n_decay = 0;
    for (const TensorDesc& t : L.tensors) {
        const bool want = t.name == "wte" || (t.ndim == 2 && t.name != "wpe");
        if (t.decay != want) test::report(__FILE__, __LINE__, "decay flag wrong for " + t.name);
        n_decay += t.decay;
    }
    EXPECT_EQ(n_decay, 1 + 4 * m.n_layer);

    // Untied adds head_w [V, C] with decay, at the end.
    const ParamLayout U = make_param_layout(preset("configs/small.cfg", 0));
    EXPECT_TRUE(!U.tied);
    ASSERT_TRUE(U.head_w >= 0);
    EXPECT_TRUE(U.head_w != U.wte);
    EXPECT_EQ(U[U.head_w].name, std::string("head_w"));
    EXPECT_EQ(U[U.head_w].shape[0], V);
    EXPECT_TRUE(U[U.head_w].decay);
    EXPECT_EQ(U.head_w, int(U.tensors.size()) - 1);
}

TEST(layouts_are_deterministic) {
    const ModelConfig m = preset("configs/base.cfg", 1);
    const ParamLayout a = make_param_layout(m), b = make_param_layout(m);
    ASSERT_EQ(a.tensors.size(), b.tensors.size());
    for (size_t i = 0; i < a.tensors.size(); ++i) {
        EXPECT_EQ(a.tensors[i].offset, b.tensors[i].offset);
        EXPECT_EQ(a.tensors[i].name, b.tensors[i].name);
    }
    const ActLayout x = make_act_layout(m, 4, 64), y = make_act_layout(m, 4, 64);
    EXPECT_EQ(x.num_elems, y.num_elems);
}

TEST(act_layout_structure_and_sizes) {
    const char* paths[] = {"configs/smoke.cfg", "configs/small.cfg", "configs/base.cfg"};
    for (const char* p : paths) {
        Config c;
        std::string err;
        ASSERT_TRUE(config_load_file(p, &c, &err));
        const ModelConfig& m = c.model;
        const int B = c.train.batch_size, T = c.train.seq_len;
        const ActLayout A = make_act_layout(m, B, T);
        check_structure(A.tensors, A.num_elems, p);
        EXPECT_EQ(A.B, B);
        EXPECT_EQ(A.T, T);

        // Expected logical element count from the §6.2 tensor list.
        const size_t N = size_t(B) * T, C = size_t(m.n_embd), V = size_t(m.vocab_size);
        const size_t att = size_t(B) * m.n_head * T * T;
        const size_t per_layer = 18 * N * C + 4 * N + att;  // see layout.h list
        const size_t want = N * C + m.n_layer * per_layer + N * C + 2 * N + 2 * N * V + N;
        size_t sum = 0;
        for (const TensorDesc& t : A.tensors) sum += t.numel;
        EXPECT_EQ(sum, want);

        const LayerActIdx& x = A.layers.back();
        EXPECT_EQ(A[x.att].ndim, 4);
        EXPECT_EQ(A[x.att].shape[1], m.n_head);
        EXPECT_EQ(A[x.qkv].shape[1], 3 * m.n_embd);
        EXPECT_EQ(A[A.logits].shape[1], m.vocab_size);
        EXPECT_EQ(A[A.probs].numel, N * V);
        EXPECT_EQ(A[x.ln1_rstd].numel, N);
        std::printf("    %-18s B=%d T=%d  activation arena %zu elems = %.1f MB (float)\n", p, B, T,
                    A.num_elems, A.num_elems * 4.0 / (1024.0 * 1024.0));
    }
}

TEST(invalid_arguments_abort) {
    std::string err;
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            ModelConfig m;
            make_act_layout(m, 1, m.max_seq_len + 1);
        },
        &err));
    EXPECT_TRUE(err.find("max_seq_len") != std::string::npos);
    err.clear();
    EXPECT_TRUE(test::dies_with_abort(
        [] {
            ModelConfig m;
            m.n_head = 3;
            make_param_layout(m);
        },
        &err));
    EXPECT_TRUE(err.find("divisible by n_head") != std::string::npos);
    err.clear();
    EXPECT_TRUE(test::dies_with_abort([] { make_act_layout(ModelConfig{}, 0, 8); }, &err));
}

TEST_MAIN()
