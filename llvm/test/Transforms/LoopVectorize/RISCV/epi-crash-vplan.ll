; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+zepi -riscv-v-vector-bits-min=64 \
; RUN:    -scalable-vectorization=only \
; RUN:    -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN:    -S -O2 < %s  -o - | FileCheck %s --check-prefix=CHECK

; ModuleID = 'crash.c'
source_filename = "crash.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: argmemonly nofree norecurse nosync nounwind writeonly
define dso_local void @foo(ptr nocapture noundef readnone %A, ptr nocapture noundef writeonly %B, i32 noundef signext %TC) local_unnamed_addr #0 {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP5_NOT:%.*]] = icmp eq i32 [[TC:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP5_NOT]], label [[FOR_COND_CLEANUP:%.*]], label [[FOR_BODY_PREHEADER:%.*]]
; CHECK:       for.body.preheader:
; CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[TC]] to i64
; CHECK-NEXT:    [[TMP0:%.*]] = tail call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP1:%.*]] = tail call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP2:%.*]] = add i32 [[TMP1]], -1
; CHECK-NEXT:    [[VECTOR_RECUR_INIT:%.*]] = insertelement <vscale x 1 x i32> poison, i32 33, i32 [[TMP2]]
; CHECK-NEXT:    [[TMP3:%.*]] = tail call <vscale x 1 x i32> @llvm.experimental.stepvector.nxv1i32()
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[FOR_BODY_PREHEADER]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[PREV_EVL:%.*]] = phi i32 [ [[TMP0]], [[FOR_BODY_PREHEADER]] ], [ [[TMP6:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <vscale x 1 x i32> [ [[VECTOR_RECUR_INIT]], [[FOR_BODY_PREHEADER]] ], [ [[VEC_IND6:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND6]] = phi <vscale x 1 x i32> [ [[TMP3]], [[FOR_BODY_PREHEADER]] ], [ [[VEC_IND_NEXT9:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[INDEX]]
; CHECK-NEXT:    [[TMP5:%.*]] = tail call i64 @llvm.epi.vsetvl(i64 [[TMP4]], i64 2, i64 7)
; CHECK-NEXT:    [[TMP6]] = trunc i64 [[TMP5]] to i32
; CHECK-NEXT:    [[DOTSPLATINSERT7:%.*]] = insertelement <vscale x 1 x i32> poison, i32 [[TMP6]], i64 0
; CHECK-NEXT:    [[DOTSPLAT8:%.*]] = shufflevector <vscale x 1 x i32> [[DOTSPLATINSERT7]], <vscale x 1 x i32> poison, <vscale x 1 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = add i32 [[PREV_EVL]], -1
; CHECK-NEXT:    [[TMP8:%.*]] = tail call <vscale x 1 x i32> @llvm.experimental.vp.splice.nxv1i32(<vscale x 1 x i32> [[VECTOR_RECUR]], <vscale x 1 x i32> [[VEC_IND6]], i32 [[TMP7]], <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> poison, i1 true, i64 0), <vscale x 1 x i1> poison, <vscale x 1 x i32> zeroinitializer), i32 [[PREV_EVL]], i32 [[TMP6]])
; CHECK-NEXT:    [[VP_OP:%.*]] = tail call <vscale x 1 x i32> @llvm.vp.add.nxv1i32(<vscale x 1 x i32> [[TMP8]], <vscale x 1 x i32> [[VEC_IND6]], <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> poison, i1 true, i64 0), <vscale x 1 x i1> poison, <vscale x 1 x i32> zeroinitializer), i32 [[TMP6]])
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i32, ptr [[B:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    tail call void @llvm.vp.store.nxv1i32.p0(<vscale x 1 x i32> [[VP_OP]], ptr [[TMP9]], <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> poison, i1 true, i64 0), <vscale x 1 x i1> poison, <vscale x 1 x i32> zeroinitializer), i32 [[TMP6]]), !tbaa [[TBAA4:![0-9]+]]
; CHECK-NEXT:    [[TMP10:%.*]] = and i64 [[TMP5]], 4294967295
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP10]]
; CHECK-NEXT:    [[VEC_IND_NEXT9]] = add <vscale x 1 x i32> [[VEC_IND6]], [[DOTSPLAT8]]
; CHECK-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[WIDE_TRIP_COUNT]]
; CHECK-NEXT:    br i1 [[TMP11]], label [[FOR_COND_CLEANUP]], label [[VECTOR_BODY]], !llvm.loop [[LOOP8:![0-9]+]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
entry:
  %cmp5.not = icmp eq i32 %TC, 0
  br i1 %cmp5.not, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %TC to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %for1.06 = phi i32 [ 33, %for.body.preheader ], [ %0, %for.body ]
  %0 = trunc i64 %indvars.iv to i32
  %add = add i32 %for1.06, %0
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  store i32 %add, ptr %arrayidx, align 4, !tbaa !4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !8
}

attributes #0 = { argmemonly nofree norecurse nosync nounwind writeonly "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+a,+c,+d,+f,+m,+zepi,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl32b,+zvl64b,-relax,-save-restore" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64d"}
!2 = !{i32 1, !"SmallDataLimit", i32 8}
!3 = !{!"clang version 16.0.0"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!"llvm.loop.vectorize.enable", i1 true}
