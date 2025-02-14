; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+m,+f,+d,+v -verify-machineinstrs < %s | FileCheck %s

define <vscale x 1 x double> @test_load_vp_reverse_nxv1f64(<vscale x 1 x double> *%ptr, i32 zeroext %evl) {
; CHECK-LABEL: test_load_vp_reverse_nxv1f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    slli a2, a1, 3
; CHECK-NEXT:    add a0, a2, a0
; CHECK-NEXT:    addi a0, a0, -8
; CHECK-NEXT:    li a2, -8
; CHECK-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-NEXT:    vlse64.v v8, (a0), a2
; CHECK-NEXT:    ret
  %head = insertelement <vscale x 1 x i1> undef, i1 1, i32 0
  %allones = shufflevector <vscale x 1 x i1> %head, <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer

  %v = call <vscale x 1 x double> @llvm.vp.load.nxv1f64(<vscale x 1 x double>* %ptr, <vscale x 1 x i1> %allones, i32 %evl)
  %dst = call <vscale x 1 x double> @llvm.experimental.vp.reverse.nxv1f64(<vscale x 1 x double> %v, <vscale x 1 x i1> %allones, i32 %evl)
  ret <vscale x 1 x double> %dst
}

define <vscale x 1 x double> @test_load_vp_reverse_different_evl_nxv1f64(<vscale x 1 x double> *%ptr, i32 zeroext %evl1, i32 zeroext %evl2) {
; CHECK-LABEL: test_load_vp_reverse_different_evl_nxv1f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-NEXT:    vle64.v v9, (a0)
; CHECK-NEXT:    vsetvli zero, a2, e64, m1, ta, ma
; CHECK-NEXT:    vid.v v8
; CHECK-NEXT:    addi a2, a2, -1
; CHECK-NEXT:    vrsub.vx v10, v8, a2
; CHECK-NEXT:    vrgather.vv v8, v9, v10
; CHECK-NEXT:    ret
  %head = insertelement <vscale x 1 x i1> undef, i1 1, i32 0
  %allones = shufflevector <vscale x 1 x i1> %head, <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer

  %v = call <vscale x 1 x double> @llvm.vp.load.nxv1f64(<vscale x 1 x double>* %ptr, <vscale x 1 x i1> %allones, i32 %evl1)
  %dst = call <vscale x 1 x double> @llvm.experimental.vp.reverse.nxv1f64(<vscale x 1 x double> %v, <vscale x 1 x i1> %allones, i32 %evl2)
  ret <vscale x 1 x double> %dst
}

define <vscale x 1 x double> @test_load_vp_reverse_many_uses_nxv1f64(<vscale x 1 x double> *%ptr, i32 zeroext %evl) {
; CHECK-LABEL: test_load_vp_reverse_many_uses_nxv1f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-NEXT:    vle64.v v8, (a0)
; CHECK-NEXT:    vid.v v9
; CHECK-NEXT:    addi a1, a1, -1
; CHECK-NEXT:    vrsub.vx v9, v9, a1
; CHECK-NEXT:    vrgather.vv v10, v8, v9
; CHECK-NEXT:    vfadd.vv v8, v8, v10
; CHECK-NEXT:    ret
  %head = insertelement <vscale x 1 x i1> undef, i1 1, i32 0
  %allones = shufflevector <vscale x 1 x i1> %head, <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer

  %v = call <vscale x 1 x double> @llvm.vp.load.nxv1f64(<vscale x 1 x double>* %ptr, <vscale x 1 x i1> %allones, i32 %evl)
  %rev = call <vscale x 1 x double> @llvm.experimental.vp.reverse.nxv1f64(<vscale x 1 x double> %v, <vscale x 1 x i1> %allones, i32 %evl)
  %z = call <vscale x 1 x double> @llvm.vp.fadd.nxv1f64(<vscale x 1 x double> %v, <vscale x 1 x double> %rev, <vscale x 1 x i1> %allones, i32 %evl)
  ret <vscale x 1 x double> %z
}

define <vscale x 1 x double> @test_load_vp_reverse_general_mask_nxv1f64(<vscale x 1 x double> *%ptr, <vscale x 1 x i1> %mask, i32 zeroext %evl) {
; CHECK-LABEL: test_load_vp_reverse_general_mask_nxv1f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-NEXT:    vle64.v v9, (a0), v0.t
; CHECK-NEXT:    vid.v v8
; CHECK-NEXT:    addi a1, a1, -1
; CHECK-NEXT:    vrsub.vx v10, v8, a1
; CHECK-NEXT:    vrgather.vv v8, v9, v10
; CHECK-NEXT:    ret
  %head = insertelement <vscale x 1 x i1> undef, i1 1, i32 0
  %allones = shufflevector <vscale x 1 x i1> %head, <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer

  %v = call <vscale x 1 x double> @llvm.vp.load.nxv1f64(<vscale x 1 x double>* %ptr, <vscale x 1 x i1> %mask, i32 %evl)
  %dst = call <vscale x 1 x double> @llvm.experimental.vp.reverse.nxv1f64(<vscale x 1 x double> %v, <vscale x 1 x i1> %allones, i32 %evl)
  ret <vscale x 1 x double> %dst
}


define <vscale x 1 x double> @test_load_vp_reverse_mask_nxv1f64(<vscale x 1 x double> *%ptr, <vscale x 1 x i1> %mask, i32 zeroext %evl) {
; CHECK-LABEL: test_load_vp_reverse_mask_nxv1f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    slli a2, a1, 3
; CHECK-NEXT:    add a0, a2, a0
; CHECK-NEXT:    addi a0, a0, -8
; CHECK-NEXT:    li a2, -8
; CHECK-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-NEXT:    vlse64.v v8, (a0), a2, v0.t
; CHECK-NEXT:    ret
  %head = insertelement <vscale x 1 x i1> undef, i1 1, i32 0
  %allones = shufflevector <vscale x 1 x i1> %head, <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer

  %rev.mask = call <vscale x 1 x i1> @llvm.experimental.vp.reverse.nxv1i1(<vscale x 1 x i1> %mask, <vscale x 1 x i1> %allones, i32 %evl)
  %v = call <vscale x 1 x double> @llvm.vp.load.nxv1f64(<vscale x 1 x double>* %ptr, <vscale x 1 x i1> %rev.mask, i32 %evl)
  %dst = call <vscale x 1 x double> @llvm.experimental.vp.reverse.nxv1f64(<vscale x 1 x double> %v, <vscale x 1 x i1> %allones, i32 %evl)
  ret <vscale x 1 x double> %dst
}

define <vscale x 1 x double> @test_load_vp_inconsistent_mask_nxv1f64(<vscale x 1 x double> *%ptr, <vscale x 1 x i1> %mask, <vscale x 1 x i1> %mask2, i32 zeroext %evl) {
; CHECK-LABEL: test_load_vp_inconsistent_mask_nxv1f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmv1r.v v9, v0
; CHECK-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-NEXT:    vmv1r.v v0, v8
; CHECK-NEXT:    vid.v v10, v0.t
; CHECK-NEXT:    addi a1, a1, -1
; CHECK-NEXT:    vrsub.vx v10, v10, a1, v0.t
; CHECK-NEXT:    vmv.v.i v11, 0
; CHECK-NEXT:    vmv1r.v v0, v9
; CHECK-NEXT:    vmerge.vim v9, v11, 1, v0
; CHECK-NEXT:    vmv1r.v v0, v8
; CHECK-NEXT:    vrgather.vv v11, v9, v10, v0.t
; CHECK-NEXT:    vmsne.vi v0, v11, 0, v0.t
; CHECK-NEXT:    vle64.v v9, (a0), v0.t
; CHECK-NEXT:    vid.v v8
; CHECK-NEXT:    vrsub.vx v10, v8, a1
; CHECK-NEXT:    vrgather.vv v8, v9, v10
; CHECK-NEXT:    ret
  %head = insertelement <vscale x 1 x i1> undef, i1 1, i32 0
  %allones = shufflevector <vscale x 1 x i1> %head, <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer

  %rev.mask = call <vscale x 1 x i1> @llvm.experimental.vp.reverse.nxv1i1(<vscale x 1 x i1> %mask, <vscale x 1 x i1> %mask2, i32 %evl)
  %v = call <vscale x 1 x double> @llvm.vp.load.nxv1f64(<vscale x 1 x double>* %ptr, <vscale x 1 x i1> %rev.mask, i32 %evl)
  %dst = call <vscale x 1 x double> @llvm.experimental.vp.reverse.nxv1f64(<vscale x 1 x double> %v, <vscale x 1 x i1> %allones, i32 %evl)
  ret <vscale x 1 x double> %dst
}

declare <vscale x 1 x double> @llvm.vp.load.nxv1f64(<vscale x 1 x double>*, <vscale x 1 x i1>, i32)
declare <vscale x 1 x double> @llvm.vp.fadd.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, <vscale x 1 x i1>, i32)
declare <vscale x 1 x double> @llvm.experimental.vp.reverse.nxv1f64(<vscale x 1 x double>, <vscale x 1 x i1>, i32)
declare <vscale x 1 x i1> @llvm.experimental.vp.reverse.nxv1i1(<vscale x 1 x i1>, <vscale x 1 x i1>, i32)
