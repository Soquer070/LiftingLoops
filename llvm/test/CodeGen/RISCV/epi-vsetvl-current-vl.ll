; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+d,+v -verify-machineinstrs \
; RUN:    -O0 < %s -epi-pipeline | FileCheck --check-prefix=CHECK-O0 %s
; RUN: llc -mtriple=riscv64 -mattr=+d,+v -verify-machineinstrs \
; RUN:    -O2 < %s -epi-pipeline | FileCheck --check-prefix=CHECK-O2 %s

; At RVV-0.8, the vsetvli instruction can be to change vtype without changing
; vl (vsetvli zero, zero, vtypei). We are using this for instructions that
; honour vtype but ignore VL, such as vfmv.f.s.

declare double @llvm.epi.vfmv.f.s.f64.nxv1f64(<vscale x 1 x double>)
declare float @llvm.epi.vfmv.f.s.f32.nxv2f32(<vscale x 2 x float>)

define double @test_change_vtype_only_1(<vscale x 1 x double> %a, <vscale x 2 x float> %b) nounwind {
; CHECK-O0-LABEL: test_change_vtype_only_1:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    vsetivli zero, 0, e64, m1, ta, mu
; CHECK-O0-NEXT:    vfmv.f.s fa5, v8
; CHECK-O0-NEXT:    vsetivli zero, 0, e32, m1, ta, mu
; CHECK-O0-NEXT:    vfmv.f.s fa4, v9
; CHECK-O0-NEXT:    # kill: def $f14_f killed $f14_f killed $f14_d
; CHECK-O0-NEXT:    fcvt.d.s fa4, fa4
; CHECK-O0-NEXT:    fadd.d fa0, fa5, fa4
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_change_vtype_only_1:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    vsetivli zero, 0, e64, m1, ta, mu
; CHECK-O2-NEXT:    vfmv.f.s fa5, v8
; CHECK-O2-NEXT:    vsetivli zero, 0, e32, m1, ta, mu
; CHECK-O2-NEXT:    vfmv.f.s fa4, v9
; CHECK-O2-NEXT:    fcvt.d.s fa4, fa4
; CHECK-O2-NEXT:    fadd.d fa0, fa5, fa4
; CHECK-O2-NEXT:    ret
  %1 = call double @llvm.epi.vfmv.f.s.f64.nxv1f64(<vscale x 1 x double> %a)
  %2 = call float @llvm.epi.vfmv.f.s.f32.nxv2f32(<vscale x 2 x float> %b)
  %conv = fpext float %2 to double
  %add = fadd double %1, %conv
  ret double %add
}

define double @test_change_vtype_only_2(<vscale x 1 x double> %a, <vscale x 1 x double> %b) nounwind {
; CHECK-O0-LABEL: test_change_vtype_only_2:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    vsetivli zero, 0, e64, m1, ta, mu
; CHECK-O0-NEXT:    vfmv.f.s fa5, v8
; CHECK-O0-NEXT:    vfmv.f.s fa4, v9
; CHECK-O0-NEXT:    fadd.d fa0, fa5, fa4
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_change_vtype_only_2:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    vsetivli zero, 0, e64, m1, ta, mu
; CHECK-O2-NEXT:    vfmv.f.s fa5, v8
; CHECK-O2-NEXT:    vfmv.f.s fa4, v9
; CHECK-O2-NEXT:    fadd.d fa0, fa5, fa4
; CHECK-O2-NEXT:    ret
  %1 = call double @llvm.epi.vfmv.f.s.f64.nxv1f64(<vscale x 1 x double> %a)
  %2 = call double @llvm.epi.vfmv.f.s.f64.nxv1f64(<vscale x 1 x double> %b)
  %add = fadd double %1, %2
  ret double %add
}
