; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+f,+d,+v -verify-machineinstrs -O0 \
; RUN:    < %s -epi-pipeline | FileCheck --check-prefix=CHECK-O0 %s
; RUN: llc -mtriple=riscv64 -mattr=+f,+d,+v -verify-machineinstrs -O2 \
; RUN:    < %s -epi-pipeline | FileCheck --check-prefix=CHECK-O2 %s

@scratch = global i8 0, align 16

define void @test_vp_select_int(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, <vscale x 1 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-O0-LABEL: test_vp_select_int:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    vmv1r.v v10, v8
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmerge.vvm v8, v9, v10, v0
; CHECK-O0-NEXT:    vs1r.v v8, (a0)
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_select_int:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmerge.vvm v8, v9, v8, v0
; CHECK-O2-NEXT:    vs1r.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 1 x i64>*

  %select = call <vscale x 1 x i64> @llvm.vp.select.nxv1i64(<vscale x 1 x i1> %m, <vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i32 %n)
  store <vscale x 1 x i64> %select, <vscale x 1 x i64>* %store_addr

  ret void
}

define void @test_vp_select_int_2(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, <vscale x 2 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-O0-LABEL: test_vp_select_int_2:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    vmv1r.v v10, v8
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmerge.vvm v8, v9, v10, v0
; CHECK-O0-NEXT:    vs1r.v v8, (a0)
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_select_int_2:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmerge.vvm v8, v9, v8, v0
; CHECK-O2-NEXT:    vs1r.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x i32>*

  %select = call <vscale x 2 x i32> @llvm.vp.select.nxv2i32(<vscale x 2 x i1> %m, <vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i32 %n)
  store <vscale x 2 x i32> %select, <vscale x 2 x i32>* %store_addr

  ret void
}

define void @test_vp_select_int_3(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-O0-LABEL: test_vp_select_int_3:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    vmv2r.v v12, v8
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    # implicit-def: $v8m2
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmerge.vvm v8, v10, v12, v0
; CHECK-O0-NEXT:    vs2r.v v8, (a0)
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_select_int_3:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmerge.vvm v8, v10, v8, v0
; CHECK-O2-NEXT:    vs2r.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x i64>*

  %select = call <vscale x 2 x i64> @llvm.vp.select.nxv2i64(<vscale x 2 x i1> %m, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i32 %n)
  store <vscale x 2 x i64> %select, <vscale x 2 x i64>* %store_addr

  ret void
}

define void @test_vp_select_fp(<vscale x 1 x double> %a, <vscale x 1 x double> %b, <vscale x 1 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-O0-LABEL: test_vp_select_fp:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    vmv1r.v v10, v8
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmerge.vvm v8, v9, v10, v0
; CHECK-O0-NEXT:    vs1r.v v8, (a0)
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_select_fp:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmerge.vvm v8, v9, v8, v0
; CHECK-O2-NEXT:    vs1r.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 1 x double>*

  %select = call <vscale x 1 x double> @llvm.vp.select.nxv1f64(<vscale x 1 x i1> %m, <vscale x 1 x double> %a, <vscale x 1 x double> %b, i32 %n)
  store <vscale x 1 x double> %select, <vscale x 1 x double>* %store_addr

  ret void
}

define void @test_vp_select_fp_2(<vscale x 2 x float> %a, <vscale x 2 x float> %b, <vscale x 2 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-O0-LABEL: test_vp_select_fp_2:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    vmv1r.v v10, v8
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    # implicit-def: $v8
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmerge.vvm v8, v9, v10, v0
; CHECK-O0-NEXT:    vs1r.v v8, (a0)
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_select_fp_2:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmerge.vvm v8, v9, v8, v0
; CHECK-O2-NEXT:    vs1r.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x float>*

  %select = call <vscale x 2 x float> @llvm.vp.select.nxv2f32(<vscale x 2 x i1> %m, <vscale x 2 x float> %a, <vscale x 2 x float> %b, i32 %n)
  store <vscale x 2 x float> %select, <vscale x 2 x float>* %store_addr

  ret void
}

define void @test_vp_select_fp_3(<vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-O0-LABEL: test_vp_select_fp_3:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    vmv2r.v v12, v8
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    # implicit-def: $v8m2
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmerge.vvm v8, v10, v12, v0
; CHECK-O0-NEXT:    vs2r.v v8, (a0)
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_select_fp_3:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmerge.vvm v8, v10, v8, v0
; CHECK-O2-NEXT:    vs2r.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x double>*

  %select = call <vscale x 2 x double> @llvm.vp.select.nxv2f64(<vscale x 2 x i1> %m, <vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 %n)
  store <vscale x 2 x double> %select, <vscale x 2 x double>* %store_addr

  ret void
}

define void @test_vp_select_mask(<vscale x 1 x i1> %a, <vscale x 1 x i1> %b, <vscale x 1 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-O0-LABEL: test_vp_select_mask:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    addi sp, sp, -16
; CHECK-O0-NEXT:    csrr a1, vlenb
; CHECK-O0-NEXT:    slli a1, a1, 1
; CHECK-O0-NEXT:    sub sp, sp, a1
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    addi a0, sp, 16
; CHECK-O0-NEXT:    vs1r.v v9, (a0) # Unknown-size Folded Spill
; CHECK-O0-NEXT:    vmv1r.v v9, v8
; CHECK-O0-NEXT:    addi a0, sp, 16
; CHECK-O0-NEXT:    vl1r.v v8, (a0) # Unknown-size Folded Reload
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    vsetvli zero, a1, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vmandn.mm v9, v9, v8
; CHECK-O0-NEXT:    vmand.mm v8, v0, v8
; CHECK-O0-NEXT:    vmor.mm v8, v8, v9
; CHECK-O0-NEXT:    vsetvli a1, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    csrr a0, vlenb
; CHECK-O0-NEXT:    slli a0, a0, 1
; CHECK-O0-NEXT:    add sp, sp, a0
; CHECK-O0-NEXT:    addi sp, sp, 16
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_select_mask:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli zero, a0, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vmandn.mm v8, v8, v9
; CHECK-O2-NEXT:    vmand.mm v9, v0, v9
; CHECK-O2-NEXT:    vmor.mm v8, v9, v8
; CHECK-O2-NEXT:    vsetvli a0, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 1 x i1>*

  %select = call <vscale x 1 x i1> @llvm.vp.select.nxv1i1(<vscale x 1 x i1> %m, <vscale x 1 x i1> %a, <vscale x 1 x i1> %b, i32 %n)
  store <vscale x 1 x i1> %select, <vscale x 1 x i1>* %store_addr

  ret void
}

define void @test_vp_select_mask_2(<vscale x 2 x i1> %a, <vscale x 2 x i1> %b, <vscale x 2 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-O0-LABEL: test_vp_select_mask_2:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    addi sp, sp, -16
; CHECK-O0-NEXT:    csrr a1, vlenb
; CHECK-O0-NEXT:    slli a1, a1, 1
; CHECK-O0-NEXT:    sub sp, sp, a1
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    addi a0, sp, 16
; CHECK-O0-NEXT:    vs1r.v v9, (a0) # Unknown-size Folded Spill
; CHECK-O0-NEXT:    vmv1r.v v9, v8
; CHECK-O0-NEXT:    addi a0, sp, 16
; CHECK-O0-NEXT:    vl1r.v v8, (a0) # Unknown-size Folded Reload
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    vsetvli zero, a1, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vmandn.mm v9, v9, v8
; CHECK-O0-NEXT:    vmand.mm v8, v0, v8
; CHECK-O0-NEXT:    vmor.mm v8, v8, v9
; CHECK-O0-NEXT:    vsetvli a1, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    csrr a0, vlenb
; CHECK-O0-NEXT:    slli a0, a0, 1
; CHECK-O0-NEXT:    add sp, sp, a0
; CHECK-O0-NEXT:    addi sp, sp, 16
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_select_mask_2:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli zero, a0, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vmandn.mm v8, v8, v9
; CHECK-O2-NEXT:    vmand.mm v9, v0, v9
; CHECK-O2-NEXT:    vmor.mm v8, v9, v8
; CHECK-O2-NEXT:    vsetvli a0, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x i1>*

  %select = call <vscale x 2 x i1> @llvm.vp.select.nxv2i1(<vscale x 2 x i1> %m, <vscale x 2 x i1> %a, <vscale x 2 x i1> %b, i32 %n)
  store <vscale x 2 x i1> %select, <vscale x 2 x i1>* %store_addr

  ret void
}

define void @test_vp_select_mask_3(<vscale x 2 x i1> %a, <vscale x 2 x i1> %b, <vscale x 2 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-O0-LABEL: test_vp_select_mask_3:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    addi sp, sp, -16
; CHECK-O0-NEXT:    csrr a1, vlenb
; CHECK-O0-NEXT:    slli a1, a1, 1
; CHECK-O0-NEXT:    sub sp, sp, a1
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    addi a0, sp, 16
; CHECK-O0-NEXT:    vs1r.v v9, (a0) # Unknown-size Folded Spill
; CHECK-O0-NEXT:    vmv1r.v v9, v8
; CHECK-O0-NEXT:    addi a0, sp, 16
; CHECK-O0-NEXT:    vl1r.v v8, (a0) # Unknown-size Folded Reload
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    vsetvli zero, a1, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vmandn.mm v9, v9, v8
; CHECK-O0-NEXT:    vmand.mm v8, v0, v8
; CHECK-O0-NEXT:    vmor.mm v8, v8, v9
; CHECK-O0-NEXT:    vsetvli a1, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    csrr a0, vlenb
; CHECK-O0-NEXT:    slli a0, a0, 1
; CHECK-O0-NEXT:    add sp, sp, a0
; CHECK-O0-NEXT:    addi sp, sp, 16
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_select_mask_3:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli zero, a0, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vmandn.mm v8, v8, v9
; CHECK-O2-NEXT:    vmand.mm v9, v0, v9
; CHECK-O2-NEXT:    vmor.mm v8, v9, v8
; CHECK-O2-NEXT:    vsetvli a0, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x i1>*

  %select = call <vscale x 2 x i1> @llvm.vp.select.nxv2i1(<vscale x 2 x i1> %m, <vscale x 2 x i1> %a, <vscale x 2 x i1> %b, i32 %n)
  store <vscale x 2 x i1> %select, <vscale x 2 x i1>* %store_addr

  ret void
}

; store
declare void @llvm.vp.store.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>*, i32, <vscale x 1 x i1>, i32)
declare void @llvm.vp.store.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>*, i32, <vscale x 2 x i1>, i32)
declare void @llvm.vp.store.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>*, i32, <vscale x 2 x i1>, i32)

; select
declare <vscale x 1 x i64> @llvm.vp.select.nxv1i64(<vscale x 1 x i1>, <vscale x 1 x i64>, <vscale x 1 x i64>, i32)
declare <vscale x 2 x i32> @llvm.vp.select.nxv2i32(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i32)
declare <vscale x 2 x i64> @llvm.vp.select.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>, i32)

declare <vscale x 1 x double> @llvm.vp.select.nxv1f64(<vscale x 1 x i1>, <vscale x 1 x double>, <vscale x 1 x double>, i32)
declare <vscale x 2 x float> @llvm.vp.select.nxv2f32(<vscale x 2 x i1>, <vscale x 2 x float>, <vscale x 2 x float>, i32)
declare <vscale x 2 x double> @llvm.vp.select.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>, i32)

declare <vscale x 1 x i1> @llvm.vp.select.nxv1i1(<vscale x 1 x i1>, <vscale x 1 x i1>, <vscale x 1 x i1>, i32)
declare <vscale x 2 x i1> @llvm.vp.select.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x i1>, i32)
