// test_config.cpp: common.h (real_t, CHECK) and config.h (P0.2).
// Covers: preset files match ROADMAP §2.1; unknown keys are errors in both
// files and on the CLI; strict value parsing; precedence; validation; exact
// round trip through a manifest.txt file; CHECK aborts with file/line.
#include "test.h"

#include "tslm/common.h"
#include "tslm/config.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

using namespace tslm;

// Expect `text` to be rejected, with `needle` appearing in the error message.
static void expect_text_error(const std::string& text, const std::string& needle, int line) {
    Config c;
    std::string err;
    const bool ok = config_parse_text(text, "inline", &c, &err);
    if (ok || err.find(needle) == std::string::npos) {
        test::report(__FILE__, line,
                     "expected error containing '" + needle + "', got ok=" + (ok ? "1" : "0") +
                         " err='" + err + "'");
    }
}

static bool parse_args(std::initializer_list<const char*> args, Config* c, std::string* err) {
    std::vector<const char*> argv{"prog"};
    argv.insert(argv.end(), args.begin(), args.end());
    return config_parse_args(static_cast<int>(argv.size()), argv.data(), c, err);
}

TEST(real_t) {
    EXPECT_TRUE(sizeof(real_t) == (REAL_IS_DOUBLE ? 8u : 4u));
}

TEST(presets) {
    struct Want { const char* path; int L, C, NH, T, B; double lr; };
    const Want wants[] = {
        {"configs/smoke.cfg", 1, 32, 2, 32, 4, 1e-2},  // tuned in P4.5
        {"configs/small.cfg", 2, 128, 4, 256, 32, 1e-3},
        {"configs/base.cfg", 4, 256, 8, 256, 32, 6e-4},
    };
    for (const Want& w : wants) {
        Config c;
        std::string err;
        const bool ok = config_load_file(w.path, &c, &err);
        if (!ok) std::fprintf(stderr, "load %s: %s\n", w.path, err.c_str());
        EXPECT_TRUE(ok);
        EXPECT_TRUE(config_validate(c, &err));
        EXPECT_TRUE(c.model.vocab_size == 10000);
        EXPECT_TRUE(c.model.tie_embeddings == 1);
        EXPECT_TRUE(c.model.n_layer == w.L);
        EXPECT_TRUE(c.model.n_embd == w.C);
        EXPECT_TRUE(c.model.n_head == w.NH);
        EXPECT_TRUE(c.model.max_seq_len == w.T);
        EXPECT_TRUE(c.train.seq_len == w.T);
        EXPECT_TRUE(c.train.batch_size == w.B);
        EXPECT_TRUE(c.train.lr == w.lr);
    }
}

TEST(defaults) {
    Config c;
    std::string err;
    EXPECT_TRUE(config_validate(c, &err));
    // §2.3 training defaults.
    EXPECT_TRUE(c.train.beta1 == 0.9 && c.train.beta2 == 0.95 && c.train.adam_eps == 1e-8);
    EXPECT_TRUE(c.train.weight_decay == 0.1 && c.train.grad_clip == 1.0);
    EXPECT_TRUE(c.train.warmup_steps == 200 && c.train.min_lr_frac == 0.1);
}

TEST(parse_errors) {
    expect_text_error("n_layer=2\nbogus_key=1\n", "inline:2: unknown config key 'bogus_key'", __LINE__);
    expect_text_error("n_layer\n", "expected key=value", __LINE__);
    expect_text_error("=3\n", "empty key", __LINE__);
    expect_text_error("n_layer=2\nn_layer=3\n", "duplicate key 'n_layer'", __LINE__);
    expect_text_error("n_layer=\n", "bad value", __LINE__);
    expect_text_error("n_layer=2x\n", "bad value", __LINE__);
    expect_text_error("n_layer=1.5\n", "bad value", __LINE__);
    expect_text_error("n_layer=99999999999\n", "bad value", __LINE__);
    expect_text_error("seed=-1\n", "bad value", __LINE__);
    expect_text_error("seed=+5\n", "bad value", __LINE__);
    expect_text_error("lr=abc\n", "bad value", __LINE__);
    expect_text_error("lr=nan\n", "bad value", __LINE__);
    expect_text_error("lr=inf\n", "bad value", __LINE__);

    // Comments, blank lines, CRLF, surrounding whitespace, '#' inside a value.
    Config c;
    std::string err;
    EXPECT_TRUE(config_parse_text("# c\n\n  n_layer = 3 \r\nout_dir=runs#x\n", "inline", &c, &err));
    EXPECT_TRUE(c.model.n_layer == 3);
    EXPECT_TRUE(c.train.out_dir == "runs#x");

    Config d;
    EXPECT_TRUE(!config_load_file("configs/does_not_exist.cfg", &d, &err));
    EXPECT_TRUE(err.find("cannot open") != std::string::npos);
}

TEST(cli) {
    std::string err;
    {
        // --config applies first even when it comes last; overrides win.
        Config c;
        EXPECT_TRUE(parse_args({"--n_layer", "3", "--seed", "42", "--config", "configs/small.cfg"}, &c, &err));
        EXPECT_TRUE(c.model.n_layer == 3);   // override beats file
        EXPECT_TRUE(c.model.n_embd == 128);  // from file
        EXPECT_TRUE(c.train.seed == 42u);
    }
    {
        Config c;
        EXPECT_TRUE(!parse_args({"--bogus", "1"}, &c, &err));
        EXPECT_TRUE(err.find("unknown config key 'bogus'") != std::string::npos);
    }
    {
        Config c;
        EXPECT_TRUE(!parse_args({"--n_layer"}, &c, &err));
        EXPECT_TRUE(err.find("missing value") != std::string::npos);
    }
    {
        Config c;
        EXPECT_TRUE(!parse_args({"n_layer", "2"}, &c, &err));
        EXPECT_TRUE(!parse_args({"--n_layer=2"}, &c, &err));
        EXPECT_TRUE(!parse_args({"--n_layer", "2", "--n_layer", "3"}, &c, &err));
        EXPECT_TRUE(!parse_args({"--config", "configs/small.cfg", "--config", "configs/base.cfg"}, &c, &err));
        EXPECT_TRUE(!parse_args({"--config", "configs/nope.cfg"}, &c, &err));
    }
}

TEST(validate) {
    std::string err;
    {
        Config c;
        c.model.n_head = 3;  // 256 % 3 != 0
        EXPECT_TRUE(!config_validate(c, &err));
        EXPECT_TRUE(err.find("divisible by n_head") != std::string::npos);
    }
    {
        Config c;
        c.train.seq_len = c.model.max_seq_len + 1;
        EXPECT_TRUE(!config_validate(c, &err));
    }
    {
        Config c;
        c.model.tie_embeddings = 2;
        EXPECT_TRUE(!config_validate(c, &err));
    }
    {
        Config c;
        c.train.run_name = "a/b";
        EXPECT_TRUE(!config_validate(c, &err));
    }
    {
        Config c;
        c.model.vocab_size = 70000;
        EXPECT_TRUE(!config_validate(c, &err));
    }
    {
        Config c;
        c.train.beta2 = 1.0;
        EXPECT_TRUE(!config_validate(c, &err));
    }
}

TEST(manifest_round_trip) {
    const char* tmp = std::getenv("TSLM_TEST_TMPDIR");
    const std::string path = std::string(tmp ? tmp : ".") + "/manifest.txt";

    Config a;
    std::string err;
    EXPECT_TRUE(config_load_file("configs/small.cfg", &a, &err));
    // Values without short exact decimal forms, to exercise %.17g.
    a.train.lr = 1.0 / 3.0;
    a.train.adam_eps = 1e-8 * 0.7;
    a.train.seed = 18446744073709551615ull;
    a.train.run_name = "round trip";
    a.model.tie_embeddings = 0;

    const bool saved = config_save_file(path, a, &err);
    if (!saved) std::fprintf(stderr, "save: %s\n", err.c_str());
    EXPECT_TRUE(saved);
    Config b;
    b.train.lr = 123.0;  // make sure the load overwrites everything
    const bool loaded = config_load_file(path, &b, &err);
    if (!loaded) std::fprintf(stderr, "load: %s\n", err.c_str());
    EXPECT_TRUE(loaded);
    EXPECT_TRUE(config_to_text(a) == config_to_text(b));
    EXPECT_TRUE(b.train.lr == a.train.lr);  // bit-exact
    EXPECT_TRUE(b.train.adam_eps == a.train.adam_eps);
    EXPECT_TRUE(b.train.seed == a.train.seed);
    EXPECT_TRUE(b.train.run_name == "round trip");
    EXPECT_TRUE(b.model.tie_embeddings == 0);
    std::remove(path.c_str());
}

static void failing_check() {
    const int x = 7;
    CHECK(x == 8, "x was %d", x);
}

static void die_on_unknown_cli_key() {
    const char* argv[] = {"prog", "--config", "configs/smoke.cfg", "--not_a_key", "1"};
    config_from_args_or_die(5, argv);
}

static void die_on_invalid_config() {
    const char* argv[] = {"prog", "--n_head", "3"};  // 256 % 3 != 0
    config_from_args_or_die(3, argv);
}

// CHECK must abort (SIGABRT) and name the condition, location, and message.
TEST(check_aborts) {
    std::string out;
    EXPECT_TRUE(test::dies_with_abort(failing_check, &out));
    EXPECT_TRUE(out.find("CHECK failed: x == 8") != std::string::npos);
    EXPECT_TRUE(out.find("test_config.cpp:") != std::string::npos);
    EXPECT_TRUE(out.find("x was 7") != std::string::npos);

    CHECK(1 + 1 == 2, "passing CHECK must not abort");
}

// The app entry point turns config errors into fatal errors.
TEST(from_args_or_die) {
    std::string out;
    EXPECT_TRUE(test::dies_with_abort(die_on_unknown_cli_key, &out));
    EXPECT_TRUE(out.find("unknown config key 'not_a_key'") != std::string::npos);
    out.clear();
    EXPECT_TRUE(test::dies_with_abort(die_on_invalid_config, &out));
    EXPECT_TRUE(out.find("divisible by n_head") != std::string::npos);

    const char* argv[] = {"prog", "--config", "configs/smoke.cfg", "--seed", "7"};
    const Config c = config_from_args_or_die(5, argv);
    EXPECT_TRUE(c.model.n_embd == 32 && c.train.seed == 7u);
}

TEST_MAIN()
