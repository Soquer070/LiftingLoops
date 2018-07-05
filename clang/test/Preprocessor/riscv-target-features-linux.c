// Check that when no -march is used, riscv64-unknown-linux-gnu defines
// the relevant RV64-MFDA features.
// 
// RUN: %clang -target riscv64-unknown-linux-gnu -x c -E -dM %s \
// RUN: -o - | FileCheck %s

// CHECK-NOT: __riscv_compressed
// CHECK: __riscv_atomic 1
// CHECK: __riscv_div 1
// CHECK: __riscv_fdiv 1
// CHECK: __riscv_flen 64
// CHECK: __riscv_fsqrt 1
// CHECK: __riscv_mul 1
// CHECK: __riscv_muldiv 1
