; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -target-abi lp64d -mattr=+m,+f,+d,+v \
; RUN:     -O0 < %s -verify-machineinstrs -epi-pipeline | FileCheck --check-prefix=CHECK-O0 %s
; RUN: llc -mtriple=riscv64 -target-abi lp64d -mattr=+m,+f,+d,+v \
; RUN:     -O2 < %s -verify-machineinstrs -epi-pipeline | FileCheck --check-prefix=CHECK-O2 %s

; NOTE: using volatile in order to avoid instruction selection optimizations.

@scratch = global i8 0, align 16

define void @test_vp_fold_greater_splats(<vscale x 1 x double> %a, double %b, <vscale x 1 x i1> %m, i32 %n) nounwind {
; CHECK-O0-LABEL: test_vp_fold_greater_splats:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    addi sp, sp, -16
; CHECK-O0-NEXT:    csrr a1, vlenb
; CHECK-O0-NEXT:    slli a1, a1, 1
; CHECK-O0-NEXT:    sub sp, sp, a1
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    addi a0, sp, 16
; CHECK-O0-NEXT:    vs1r.v v0, (a0) # Unknown-size Folded Spill
; CHECK-O0-NEXT:    vmv1r.v v9, v8
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    slli a1, a1, 32
; CHECK-O0-NEXT:    srli a1, a1, 32
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmfgt.vf v8, v9, fa0, v0.t
; CHECK-O0-NEXT:    addi a2, sp, 16
; CHECK-O0-NEXT:    vl1r.v v0, (a2) # Unknown-size Folded Reload
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmflt.vf v8, v9, fa0, v0.t
; CHECK-O0-NEXT:    addi a2, sp, 16
; CHECK-O0-NEXT:    vl1r.v v0, (a2) # Unknown-size Folded Reload
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmfge.vf v8, v9, fa0, v0.t
; CHECK-O0-NEXT:    addi a2, sp, 16
; CHECK-O0-NEXT:    vl1r.v v0, (a2) # Unknown-size Folded Reload
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmfle.vf v8, v9, fa0, v0.t
; CHECK-O0-NEXT:    vsetvli a1, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    csrr a0, vlenb
; CHECK-O0-NEXT:    slli a0, a0, 1
; CHECK-O0-NEXT:    add sp, sp, a0
; CHECK-O0-NEXT:    addi sp, sp, 16
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_fold_greater_splats:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    slli a0, a0, 32
; CHECK-O2-NEXT:    srli a0, a0, 32
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmfgt.vf v9, v8, fa0, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v9, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmflt.vf v9, v8, fa0, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v9, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmfge.vf v9, v8, fa0, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v9, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmfle.vf v8, v8, fa0, v0.t
; CHECK-O2-NEXT:    vsetvli a0, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 1 x i1>*

  %head = insertelement <vscale x 1 x double> undef, double %b, i32 0
  %splat = shufflevector <vscale x 1 x double> %head, <vscale x 1 x double> undef, <vscale x 1 x i32> zeroinitializer

  ; ; x > splat(y) → vmfgt.vf x, y
  %ogt.0 = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %splat, metadata !"ogt", <vscale x 1 x i1> %m, i32 %n)
  store volatile <vscale x 1 x i1> %ogt.0, <vscale x 1 x i1>* %store_addr

  ; splat(y) > x → x < splat(y) → vmflt.vf x, y
  %ogt.1 = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %splat, <vscale x 1 x double> %a, metadata !"ogt", <vscale x 1 x i1> %m, i32 %n)
  store volatile <vscale x 1 x i1> %ogt.1, <vscale x 1 x i1>* %store_addr

  ; x >= splat(y) → vmfge.vf x, y
  %oge.0 = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %splat, metadata !"oge", <vscale x 1 x i1> %m, i32 %n)
  store volatile <vscale x 1 x i1> %oge.0, <vscale x 1 x i1>* %store_addr

  ; splat(y) >= x → x <= splat(y) → vmfle.vf x, y
  %oge.1 = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %splat, <vscale x 1 x double> %a, metadata !"oge", <vscale x 1 x i1> %m, i32 %n)
  store volatile <vscale x 1 x i1> %oge.1, <vscale x 1 x i1>* %store_addr

  ret void
}

define void @test_vp_fold_lower_splats(<vscale x 1 x double> %a, <vscale x 1 x double> %b, <vscale x 1 x i1> %m, i32 %n) nounwind {
; CHECK-O0-LABEL: test_vp_fold_lower_splats:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    addi sp, sp, -16
; CHECK-O0-NEXT:    csrr a1, vlenb
; CHECK-O0-NEXT:    slli a1, a1, 1
; CHECK-O0-NEXT:    sub sp, sp, a1
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    addi a0, sp, 16
; CHECK-O0-NEXT:    vs1r.v v0, (a0) # Unknown-size Folded Spill
; CHECK-O0-NEXT:    vmv1r.v v9, v8
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    slli a1, a1, 32
; CHECK-O0-NEXT:    srli a1, a1, 32
; CHECK-O0-NEXT:    lui a2, %hi(.LCPI1_0)
; CHECK-O0-NEXT:    fld fa5, %lo(.LCPI1_0)(a2)
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmflt.vf v8, v9, fa5, v0.t
; CHECK-O0-NEXT:    addi a2, sp, 16
; CHECK-O0-NEXT:    vl1r.v v0, (a2) # Unknown-size Folded Reload
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmfgt.vf v8, v9, fa5, v0.t
; CHECK-O0-NEXT:    addi a2, sp, 16
; CHECK-O0-NEXT:    vl1r.v v0, (a2) # Unknown-size Folded Reload
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmfle.vf v8, v9, fa5, v0.t
; CHECK-O0-NEXT:    addi a2, sp, 16
; CHECK-O0-NEXT:    vl1r.v v0, (a2) # Unknown-size Folded Reload
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmfge.vf v8, v9, fa5, v0.t
; CHECK-O0-NEXT:    vsetvli a1, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    csrr a0, vlenb
; CHECK-O0-NEXT:    slli a0, a0, 1
; CHECK-O0-NEXT:    add sp, sp, a0
; CHECK-O0-NEXT:    addi sp, sp, 16
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_fold_lower_splats:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    lui a2, %hi(.LCPI1_0)
; CHECK-O2-NEXT:    fld fa5, %lo(.LCPI1_0)(a2)
; CHECK-O2-NEXT:    slli a0, a0, 32
; CHECK-O2-NEXT:    srli a0, a0, 32
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmflt.vf v9, v8, fa5, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v9, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmfgt.vf v9, v8, fa5, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v9, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmfle.vf v9, v8, fa5, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v9, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmfge.vf v8, v8, fa5, v0.t
; CHECK-O2-NEXT:    vsetvli a0, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 1 x i1>*

  %head = insertelement <vscale x 1 x double> undef, double 2.0, i32 0
  %splat = shufflevector <vscale x 1 x double> %head, <vscale x 1 x double> undef, <vscale x 1 x i32> zeroinitializer

  ; x < splat(y) → vmflt.vf x, y
  %olt.0 = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %splat, metadata !"olt", <vscale x 1 x i1> %m, i32 %n)
  store volatile <vscale x 1 x i1> %olt.0, <vscale x 1 x i1>* %store_addr

  ; splat(y) < x → x > splat(y) → vmfgt.vf x, y
  %olt.1 = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %splat, <vscale x 1 x double> %a, metadata !"olt", <vscale x 1 x i1> %m, i32 %n)
  store volatile <vscale x 1 x i1> %olt.1, <vscale x 1 x i1>* %store_addr

  ; x <= splat(y) → vmfle.vf x, y
  %ole.0 = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %splat, metadata !"ole", <vscale x 1 x i1> %m, i32 %n)
  store volatile <vscale x 1 x i1> %ole.0, <vscale x 1 x i1>* %store_addr

  ; splat(y) <= x → x >= splat(y) → vmfge.vf x, y
  %ole.1 = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %splat, <vscale x 1 x double> %a, metadata !"ole", <vscale x 1 x i1> %m, i32 %n)
  store volatile <vscale x 1 x i1> %ole.1, <vscale x 1 x i1>* %store_addr

  ret void
}

declare <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, metadata, <vscale x 1 x i1>, i32)

