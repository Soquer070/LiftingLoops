; RUN: opt -S -passes=vec-clone-vp %s | FileCheck %s --check-prefix=PARALLEL_ACCESSES
; RUN: opt -S -passes=vec-clone-vp,loop-vectorize %s | FileCheck %s --check-prefix=MEM_CHECKS

; ModuleID = 'crash_370.c'
source_filename = "crash_370.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) uwtable vscale_range(1,1024)
define dso_local void @foo(i32 noundef signext %X, i32 noundef signext %Y, ptr nocapture noundef %W, ptr nocapture noundef %Z) local_unnamed_addr #0 {
entry:
  %0 = and i32 %X, 1
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %1 = load float, ptr %W, align 4, !tbaa !8
  %conv = sitofp i32 %X to float
  %conv1 = sitofp i32 %Y to float
  %2 = tail call float @llvm.fmuladd.f32(float %1, float %conv, float %conv1)
  store float %2, ptr %W, align 4, !tbaa !8
  br label %if.end

if.else:                                          ; preds = %entry
  %3 = load double, ptr %Z, align 8, !tbaa !12
  %conv2 = sitofp i32 %X to double
  %conv3 = sitofp i32 %Y to double
  %4 = tail call double @llvm.fmuladd.f64(double %3, double %conv2, double %conv3)
  store double %4, ptr %Z, align 8, !tbaa !12
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; PARALLEL_ACCESSES-LABEL: @_ZGVENk2vvl4l8_foo(
; PARALLEL_ACCESSES:       {{.*}} = load i32, {{.*}}, !llvm.access.group [[ACCESS_GROUP:!.*]]
; PARALLEL_ACCESSES:       {{.*}} = load float, {{.*}}, !llvm.access.group [[ACCESS_GROUP]]
; PARALLEL_ACCESSES:       {{.*}} = load i32, {{.*}}, !llvm.access.group [[ACCESS_GROUP]]
; PARALLEL_ACCESSES:       {{.*}} = load i32, {{.*}}, !llvm.access.group [[ACCESS_GROUP]]
; PARALLEL_ACCESSES:       store float {{.*}}, !llvm.access.group [[ACCESS_GROUP]]
; PARALLEL_ACCESSES:       {{.*}} = load double, {{.*}}, !llvm.access.group [[ACCESS_GROUP]]
; PARALLEL_ACCESSES:       {{.*}} = load i32, {{.*}}, !llvm.access.group [[ACCESS_GROUP]]
; PARALLEL_ACCESSES:       {{.*}} = load i32, {{.*}}, !llvm.access.group [[ACCESS_GROUP]]
; PARALLEL_ACCESSES:       store double {{.*}}, !llvm.access.group [[ACCESS_GROUP]]

; MEM_CHECKS-LABEL: @_ZGVENk2vvl4l8_foo(
; MEM_CHECKS-NOT:   vector.memcheck:

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) uwtable vscale_range(1,1024) "_ZGVENk2vvl4l8_foo" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+m,+zepi,+zicsr,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl32b,+zvl64b,-e,-experimental-smaia,-experimental-ssaia,-experimental-zca,-experimental-zcb,-experimental-zcd,-experimental-zcf,-experimental-zcmp,-experimental-zcmt,-experimental-zfa,-experimental-zicond,-experimental-zihintntl,-experimental-ztso,-experimental-zvbb,-experimental-zvbc,-experimental-zvfh,-experimental-zvkg,-experimental-zvkn,-experimental-zvkned,-experimental-zvkng,-experimental-zvknha,-experimental-zvknhb,-experimental-zvks,-experimental-zvksed,-experimental-zvksg,-experimental-zvksh,-experimental-zvkt,-h,-relax,-save-restore,-svinval,-svnapot,-svpbmt,-v,-xsfvcp,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zdinx,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zicbom,-zicbop,-zicboz,-zicntr,-zifencei,-zihintpause,-zihpm,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-zmmul,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl4096b,-zvl512b,-zvl65536b,-zvl8192b" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64d"}
!2 = !{i32 7, !"openmp", i32 50}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 7, !"PIE Level", i32 2}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 8, !"SmallDataLimit", i32 8}
!7 = !{!"clang version 17.0.0"}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!13, !13, i64 0}
!13 = !{!"double", !10, i64 0}
