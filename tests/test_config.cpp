// test_config.cpp: common.h (real_t, CHECK) and config.h (P0.2).
// Covers: preset files match ROADMAP §2.1; unknown keys are errors in both
// files and on the CLI; strict value parsing; precedence; validation; exact
// round trip through a manifest.txt file; CHECK aborts with file/line.
// Uses a plain main() returning nonzero on failure (tests/test.h is P0.3).
#include "tslm/common.h"
#include "tslm/config.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <initializer_list>
#include <vector>
#include <unistd.h>

using namespace tslm;

static int g_failures = 0;

#define EXPECT(cond)                                                          \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

// Expect `text` to be rejected, with `needle` appearing in the error message.
static void expect_text_error(const std::string& text, const std::string& needle, int line) {
    Config c;
    std::string err;
    const bool ok = config_parse_text(text, "inline", &c, &err);
    if (ok || err.find(needle) == std::string::npos) {
        std::fprintf(stderr, "FAIL line %d: expected error containing '%s', got ok=%d err='%s'\n",
                     line, needle.c_str(), ok, err.c_str());
        ++g_failures;
    }
}

static bool parse_args(std::initializer_list<const char*> args, Config* c, std::string* err) {
    std::vector<const char*> argv{"prog"};
    argv.insert(argv.end(), args.begin(), args.end());
    return config_parse_args(static_cast<int>(argv.size()), argv.data(), c, err);
}

static void test_real_t() {
    EXPECT(sizeof(real_t) == (REAL_IS_DOUBLE ? 8u : 4u));
}

static void test_presets() {
    struct Want { const char* path; int L, C, NH, T, B; double lr; };
    const Want wants[] = {
        {"configs/smoke.cfg", 1, 32, 2, 32, 4, 1e-3},
        {"configs/small.cfg", 2, 128, 4, 256, 32, 1e-3},
        {"configs/base.cfg", 4, 256, 8, 256, 32, 6e-4},
    };
    for (const Want& w : wants) {
        Config c;
        std::string err;
        const bool ok = config_load_file(w.path, &c, &err);
        if (!ok) std::fprintf(stderr, "load %s: %s\n", w.path, err.c_str());
        EXPECT(ok);
        EXPECT(config_validate(c, &err));
        EXPECT(c.model.vocab_size == 10000);
        EXPECT(c.model.tie_embeddings == 1);
        EXPECT(c.model.n_layer == w.L);
        EXPECT(c.model.n_embd == w.C);
        EXPECT(c.model.n_head == w.NH);
        EXPECT(c.model.max_seq_len == w.T);
        EXPECT(c.train.seq_len == w.T);
        EXPECT(c.train.batch_size == w.B);
        EXPECT(c.train.lr == w.lr);
    }
}

static void test_defaults() {
    Config c;
    std::string err;
    EXPECT(config_validate(c, &err));
    // §2.3 training defaults.
    EXPECT(c.train.beta1 == 0.9 && c.train.beta2 == 0.95 && c.train.adam_eps == 1e-8);
    EXPECT(c.train.weight_decay == 0.1 && c.train.grad_clip == 1.0);
    EXPECT(c.train.warmup_steps == 200 && c.train.min_lr_frac == 0.1);
}

static void test_parse_errors() {
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
    EXPECT(config_parse_text("# c\n\n  n_layer = 3 \r\nout_dir=runs#x\n", "inline", &c, &err));
    EXPECT(c.model.n_layer == 3);
    EXPECT(c.train.out_dir == "runs#x");

    Config d;
    EXPECT(!config_load_file("configs/does_not_exist.cfg", &d, &err));
    EXPECT(err.find("cannot open") != std::string::npos);
}

static void test_cli() {
    std::string err;
    {
        // --config applies first even when it comes last; overrides win.
        Config c;
        EXPECT(parse_args({"--n_layer", "3", "--seed", "42", "--config", "configs/small.cfg"}, &c, &err));
        EXPECT(c.model.n_layer == 3);   // override beats file
        EXPECT(c.model.n_embd == 128);  // from file
        EXPECT(c.train.seed == 42u);
    }
    {
        Config c;
        EXPECT(!parse_args({"--bogus", "1"}, &c, &err));
        EXPECT(err.find("unknown config key 'bogus'") != std::string::npos);
    }
    {
        Config c;
        EXPECT(!parse_args({"--n_layer"}, &c, &err));
        EXPECT(err.find("missing value") != std::string::npos);
    }
    {
        Config c;
        EXPECT(!parse_args({"n_layer", "2"}, &c, &err));
        EXPECT(!parse_args({"--n_layer=2"}, &c, &err));
        EXPECT(!parse_args({"--n_layer", "2", "--n_layer", "3"}, &c, &err));
        EXPECT(!parse_args({"--config", "configs/small.cfg", "--config", "configs/base.cfg"}, &c, &err));
        EXPECT(!parse_args({"--config", "configs/nope.cfg"}, &c, &err));
    }
}

static void test_validate() {
    std::string err;
    {
        Config c;
        c.model.n_head = 3;  // 256 % 3 != 0
        EXPECT(!config_validate(c, &err));
        EXPECT(err.find("divisible by n_head") != std::string::npos);
    }
    {
        Config c;
        c.train.seq_len = c.model.max_seq_len + 1;
        EXPECT(!config_validate(c, &err));
    }
    {
        Config c;
        c.model.tie_embeddings = 2;
        EXPECT(!config_validate(c, &err));
    }
    {
        Config c;
        c.train.run_name = "a/b";
        EXPECT(!config_validate(c, &err));
    }
    {
        Config c;
        c.model.vocab_size = 70000;
        EXPECT(!config_validate(c, &err));
    }
    {
        Config c;
        c.train.beta2 = 1.0;
        EXPECT(!config_validate(c, &err));
    }
}

static void test_manifest_round_trip() {
    const char* tmp = std::getenv("TSLM_TEST_TMPDIR");
    const std::string path = std::string(tmp ? tmp : ".") + "/manifest.txt";

    Config a;
    std::string err;
    EXPECT(config_load_file("configs/small.cfg", &a, &err));
    // Values without short exact decimal forms, to exercise %.17g.
    a.train.lr = 1.0 / 3.0;
    a.train.adam_eps = 1e-8 * 0.7;
    a.train.seed = 18446744073709551615ull;
    a.train.run_name = "round trip";
    a.model.tie_embeddings = 0;

    const bool saved = config_save_file(path, a, &err);
    if (!saved) std::fprintf(stderr, "save: %s\n", err.c_str());
    EXPECT(saved);
    Config b;
    b.train.lr = 123.0;  // make sure the load overwrites everything
    const bool loaded = config_load_file(path, &b, &err);
    if (!loaded) std::fprintf(stderr, "load: %s\n", err.c_str());
    EXPECT(loaded);
    EXPECT(config_to_text(a) == config_to_text(b));
    EXPECT(b.train.lr == a.train.lr);  // bit-exact
    EXPECT(b.train.adam_eps == a.train.adam_eps);
    EXPECT(b.train.seed == a.train.seed);
    EXPECT(b.train.run_name == "round trip");
    EXPECT(b.model.tie_embeddings == 0);
    std::remove(path.c_str());
}

// Run fn() in a child process; return true if it died by SIGABRT. The
// child's stderr is captured into *out.
static bool dies_with_abort(void (*fn)(), std::string* out) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return false;
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]);
        fn();
        _exit(0);  // fn returned: no abort
    }
    close(pipefd[1]);
    char buf[512];
    ssize_t n;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) out->append(buf, static_cast<size_t>(n));
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
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
static void test_check_aborts() {
    std::string out;
    EXPECT(dies_with_abort(failing_check, &out));
    EXPECT(out.find("CHECK failed: x == 8") != std::string::npos);
    EXPECT(out.find("test_config.cpp:") != std::string::npos);
    EXPECT(out.find("x was 7") != std::string::npos);

    CHECK(1 + 1 == 2, "passing CHECK must not abort");
}

// The app entry point turns config errors into fatal errors.
static void test_from_args_or_die() {
    std::string out;
    EXPECT(dies_with_abort(die_on_unknown_cli_key, &out));
    EXPECT(out.find("unknown config key 'not_a_key'") != std::string::npos);
    out.clear();
    EXPECT(dies_with_abort(die_on_invalid_config, &out));
    EXPECT(out.find("divisible by n_head") != std::string::npos);

    const char* argv[] = {"prog", "--config", "configs/smoke.cfg", "--seed", "7"};
    const Config c = config_from_args_or_die(5, argv);
    EXPECT(c.model.n_embd == 32 && c.train.seed == 7u);
}

int main() {
    test_real_t();
    test_presets();
    test_defaults();
    test_parse_errors();
    test_cli();
    test_validate();
    test_manifest_round_trip();
    test_check_aborts();
    test_from_args_or_die();
    std::printf("test_config: %s (%d failure%s)\n", g_failures ? "FAILED" : "ok", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
