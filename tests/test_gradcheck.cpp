// test_gradcheck.cpp: full-model gradient check (ROADMAP P3.6, §10).
// DoD: all tensors pass tolerance; the test prints the worst relative error
// per tensor.
//
// REAL=double build: the §10 criterion as amended in docs/DECISIONS.md
// (5-point stencil h = 3e-4 on init weights x3, rel err < 1e-6, floor 1e-8,
// key-bias ratio <= 1e-12), on the §10 tiny model, tied and untied.
// float build: smoke run of the same check with constants float can meet
// (h = 1e-2, tol 2e-2, floor 1e-2, key-bias ratio 1e-5). It is not the DoD.
#include "test.h"

#include "tslm/common.h"
#include "tslm/gradcheck.h"

#include <cstdio>
#include <string>

using namespace tslm;

namespace {

GradcheckOptions options(int tie) {
    GradcheckOptions o = default_gradcheck_options();
    o.cfg.tie_embeddings = tie;
    if (!REAL_IS_DOUBLE) {
        o.h = 1e-2;
        o.tol = 2e-2;
        o.floor = 1e-2;
        o.kbias_ratio = 1e-5;
    }
    return o;
}

void run_case(int tie) {
    const GradcheckOptions o = options(tie);
    const GradcheckReport r = run_gradcheck(o);
    std::printf("%s", format_gradcheck_report(o, r).c_str());
    EXPECT_EQ(r.tensors.size(), size_t(tie ? 28 : 29));  // every parameter tensor checked
    for (const TensorCheck& t : r.tensors) {
        if (!t.pass) test::report(__FILE__, __LINE__, "gradcheck failed for tensor " + t.name);
        EXPECT_TRUE(t.checked >= 2);  // at least first and last index
    }
    EXPECT_TRUE(r.kbias_pass);
    EXPECT_TRUE(r.pass);
}

}  // namespace

TEST(gradcheck_tiny_tied) { run_case(1); }

TEST(gradcheck_tiny_untied) { run_case(0); }

TEST_MAIN()
