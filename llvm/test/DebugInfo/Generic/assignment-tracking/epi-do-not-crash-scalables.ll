; We only check that we do not crash with scalable vectors
; RUN: opt -passes='declare-to-assign,verify' %s -disable-output
; REQUIRES: riscv-registered-target

; ModuleID = 'addr2line-01-eda034.c'
source_filename = "addr2line-01-eda034.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

@a = dso_local global [64 x i64] zeroinitializer, align 8, !dbg !0
@b = dso_local global [64 x i64] zeroinitializer, align 8, !dbg !28
@ref = dso_local global [64 x i64] zeroinitializer, align 8, !dbg !36
@test = dso_local global [64 x i64] zeroinitializer, align 8, !dbg !34
@.str = private unnamed_addr constant [15 x i8] c"Invalid result\00", align 1, !dbg !5
@.str.1 = private unnamed_addr constant [38 x i8] c"ref[i] == test[i] && \22Invalid result\22\00", align 1, !dbg !12
@.str.2 = private unnamed_addr constant [61 x i8] c"/home/rferrer/vehave/vehave-src/tests/paraver/addr2line-01.c\00", align 1, !dbg !17
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1, !dbg !22

; Function Attrs: nounwind uwtable vscale_range(1,1024)
define dso_local signext i32 @main(i32 noundef signext %argc, ptr noundef %argv) #0 !dbg !47 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  %gvl = alloca i32, align 4
  %i = alloca i32, align 4
  %va = alloca <vscale x 1 x i64>, align 8
  %vb = alloca <vscale x 1 x i64>, align 8
  %vc = alloca <vscale x 1 x i64>, align 8
  store i32 0, ptr %retval, align 4
  store i32 %argc, ptr %argc.addr, align 4, !tbaa !64
  call void @llvm.dbg.declare(metadata ptr %argc.addr, metadata !54, metadata !DIExpression()), !dbg !68
  store ptr %argv, ptr %argv.addr, align 8, !tbaa !69
  call void @llvm.dbg.declare(metadata ptr %argv.addr, metadata !55, metadata !DIExpression()), !dbg !71
  call void @llvm.lifetime.start.p0(i64 4, ptr %gvl) #7, !dbg !72
  call void @llvm.dbg.declare(metadata ptr %gvl, metadata !56, metadata !DIExpression()), !dbg !73
  %0 = call i64 @llvm.epi.vsetvlmax(i64 3, i64 0), !dbg !74
  %conv = trunc i64 %0 to i32, !dbg !74
  store i32 %conv, ptr %gvl, align 4, !dbg !73, !tbaa !64
  call void @llvm.lifetime.start.p0(i64 4, ptr %i) #7, !dbg !75
  call void @llvm.dbg.declare(metadata ptr %i, metadata !57, metadata !DIExpression()), !dbg !76
  store i32 0, ptr %i, align 4, !dbg !77, !tbaa !64
  br label %for.cond, !dbg !79

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, ptr %i, align 4, !dbg !80, !tbaa !64
  %2 = load i32, ptr %gvl, align 4, !dbg !82, !tbaa !64
  %cmp = icmp slt i32 %1, %2, !dbg !83
  br i1 %cmp, label %for.body, label %for.end, !dbg !84

for.body:                                         ; preds = %for.cond
  %3 = load i32, ptr %i, align 4, !dbg !85, !tbaa !64
  %mul = mul nsw i32 3, %3, !dbg !87
  %conv2 = sext i32 %mul to i64, !dbg !88
  %4 = load i32, ptr %i, align 4, !dbg !89, !tbaa !64
  %idxprom = sext i32 %4 to i64, !dbg !90
  %arrayidx = getelementptr inbounds [64 x i64], ptr @a, i64 0, i64 %idxprom, !dbg !90
  store i64 %conv2, ptr %arrayidx, align 8, !dbg !91, !tbaa !92
  %5 = load i32, ptr %i, align 4, !dbg !94, !tbaa !64
  %mul3 = mul nsw i32 2, %5, !dbg !95
  %conv4 = sext i32 %mul3 to i64, !dbg !96
  %6 = load i32, ptr %i, align 4, !dbg !97, !tbaa !64
  %idxprom5 = sext i32 %6 to i64, !dbg !98
  %arrayidx6 = getelementptr inbounds [64 x i64], ptr @b, i64 0, i64 %idxprom5, !dbg !98
  store i64 %conv4, ptr %arrayidx6, align 8, !dbg !99, !tbaa !92
  %7 = load i32, ptr %i, align 4, !dbg !100, !tbaa !64
  %mul7 = mul nsw i32 5, %7, !dbg !101
  %conv8 = sext i32 %mul7 to i64, !dbg !102
  %8 = load i32, ptr %i, align 4, !dbg !103, !tbaa !64
  %idxprom9 = sext i32 %8 to i64, !dbg !104
  %arrayidx10 = getelementptr inbounds [64 x i64], ptr @ref, i64 0, i64 %idxprom9, !dbg !104
  store i64 %conv8, ptr %arrayidx10, align 8, !dbg !105, !tbaa !92
  br label %for.inc, !dbg !106

for.inc:                                          ; preds = %for.body
  %9 = load i32, ptr %i, align 4, !dbg !107, !tbaa !64
  %inc = add nsw i32 %9, 1, !dbg !107
  store i32 %inc, ptr %i, align 4, !dbg !107, !tbaa !64
  br label %for.cond, !dbg !108, !llvm.loop !109

for.end:                                          ; preds = %for.cond
  call void @llvm.lifetime.start.p0(i64 -1, ptr %va) #7, !dbg !112
  call void @llvm.dbg.declare(metadata ptr %va, metadata !58, metadata !DIExpression()), !dbg !113
  %10 = load i32, ptr %gvl, align 4, !dbg !114, !tbaa !64
  %conv11 = sext i32 %10 to i64, !dbg !114
  %11 = call <vscale x 1 x i64> @llvm.epi.vload.nxv1i64(ptr @a, i64 %conv11), !dbg !115
  store <vscale x 1 x i64> %11, ptr %va, align 8, !dbg !113, !tbaa !116
  call void @llvm.lifetime.start.p0(i64 -1, ptr %vb) #7, !dbg !117
  call void @llvm.dbg.declare(metadata ptr %vb, metadata !62, metadata !DIExpression()), !dbg !118
  %12 = load i32, ptr %gvl, align 4, !dbg !119, !tbaa !64
  %conv12 = sext i32 %12 to i64, !dbg !119
  %13 = call <vscale x 1 x i64> @llvm.epi.vload.nxv1i64(ptr @b, i64 %conv12), !dbg !120
  store <vscale x 1 x i64> %13, ptr %vb, align 8, !dbg !118, !tbaa !116
  call void @llvm.lifetime.start.p0(i64 -1, ptr %vc) #7, !dbg !121
  call void @llvm.dbg.declare(metadata ptr %vc, metadata !63, metadata !DIExpression()), !dbg !122
  %14 = load <vscale x 1 x i64>, ptr %va, align 8, !dbg !123, !tbaa !116
  %15 = load <vscale x 1 x i64>, ptr %vb, align 8, !dbg !124, !tbaa !116
  %16 = load i32, ptr %gvl, align 4, !dbg !125, !tbaa !64
  %conv13 = sext i32 %16 to i64, !dbg !125
  %17 = call <vscale x 1 x i64> @llvm.epi.vadd.nxv1i64.nxv1i64(<vscale x 1 x i64> %14, <vscale x 1 x i64> %15, i64 %conv13), !dbg !126
  store <vscale x 1 x i64> %17, ptr %vc, align 8, !dbg !122, !tbaa !116
  %18 = load <vscale x 1 x i64>, ptr %vc, align 8, !dbg !127, !tbaa !116
  %19 = load i32, ptr %gvl, align 4, !dbg !128, !tbaa !64
  %conv14 = sext i32 %19 to i64, !dbg !128
  call void @llvm.epi.vstore.nxv1i64(<vscale x 1 x i64> %18, ptr @test, i64 %conv14), !dbg !129
  store i32 0, ptr %i, align 4, !dbg !130, !tbaa !64
  br label %for.cond15, !dbg !132

for.cond15:                                       ; preds = %for.inc25, %for.end
  %20 = load i32, ptr %i, align 4, !dbg !133, !tbaa !64
  %21 = load i32, ptr %gvl, align 4, !dbg !135, !tbaa !64
  %cmp16 = icmp slt i32 %20, %21, !dbg !136
  br i1 %cmp16, label %for.body18, label %for.end27, !dbg !137

for.body18:                                       ; preds = %for.cond15
  %22 = load i32, ptr %i, align 4, !dbg !138, !tbaa !64
  %idxprom19 = sext i32 %22 to i64, !dbg !138
  %arrayidx20 = getelementptr inbounds [64 x i64], ptr @ref, i64 0, i64 %idxprom19, !dbg !138
  %23 = load i64, ptr %arrayidx20, align 8, !dbg !138, !tbaa !92
  %24 = load i32, ptr %i, align 4, !dbg !138, !tbaa !64
  %idxprom21 = sext i32 %24 to i64, !dbg !138
  %arrayidx22 = getelementptr inbounds [64 x i64], ptr @test, i64 0, i64 %idxprom21, !dbg !138
  %25 = load i64, ptr %arrayidx22, align 8, !dbg !138, !tbaa !92
  %cmp23 = icmp eq i64 %23, %25, !dbg !138
  br i1 %cmp23, label %land.lhs.true, label %if.else, !dbg !138

land.lhs.true:                                    ; preds = %for.body18
  br i1 true, label %if.then, label %if.else, !dbg !142

if.then:                                          ; preds = %land.lhs.true
  br label %if.end, !dbg !142

if.else:                                          ; preds = %land.lhs.true, %for.body18
  call void @__assert_fail(ptr noundef @.str.1, ptr noundef @.str.2, i32 noundef signext 49, ptr noundef @__PRETTY_FUNCTION__.main) #8, !dbg !138
  unreachable, !dbg !138

if.end:                                           ; preds = %if.then
  br label %for.inc25, !dbg !143

for.inc25:                                        ; preds = %if.end
  %26 = load i32, ptr %i, align 4, !dbg !144, !tbaa !64
  %inc26 = add nsw i32 %26, 1, !dbg !144
  store i32 %inc26, ptr %i, align 4, !dbg !144, !tbaa !64
  br label %for.cond15, !dbg !145, !llvm.loop !146

for.end27:                                        ; preds = %for.cond15
  call void @llvm.lifetime.end.p0(i64 -1, ptr %vc) #7, !dbg !148
  call void @llvm.lifetime.end.p0(i64 -1, ptr %vb) #7, !dbg !148
  call void @llvm.lifetime.end.p0(i64 -1, ptr %va) #7, !dbg !148
  call void @llvm.lifetime.end.p0(i64 4, ptr %i) #7, !dbg !148
  call void @llvm.lifetime.end.p0(i64 4, ptr %gvl) #7, !dbg !148
  ret i32 0, !dbg !149
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i64 @llvm.epi.vsetvlmax(i64, i64) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare <vscale x 1 x i64> @llvm.epi.vload.nxv1i64(ptr nocapture, i64) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <vscale x 1 x i64> @llvm.epi.vadd.nxv1i64.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, i64) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.epi.vstore.nxv1i64(<vscale x 1 x i64>, ptr nocapture, i64) #5

; Function Attrs: noreturn nounwind
declare !dbg !150 void @__assert_fail(ptr noundef, ptr noundef, i32 noundef signext, ptr noundef) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

attributes #0 = { nounwind uwtable vscale_range(1,1024) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+m,+zepi,+zicsr,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl32b,+zvl64b,-e,-experimental-smaia,-experimental-ssaia,-experimental-zca,-experimental-zcb,-experimental-zcd,-experimental-zcf,-experimental-zfa,-experimental-zicond,-experimental-zihintntl,-experimental-ztso,-experimental-zvbb,-experimental-zvbc,-experimental-zvfh,-experimental-zvkg,-experimental-zvkn,-experimental-zvkned,-experimental-zvkng,-experimental-zvknha,-experimental-zvknhb,-experimental-zvks,-experimental-zvksed,-experimental-zvksg,-experimental-zvksh,-experimental-zvkt,-h,-relax,-save-restore,-svinval,-svnapot,-svpbmt,-v,-xsfvcp,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zdinx,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zicbom,-zicbop,-zicboz,-zicntr,-zifencei,-zihintpause,-zihpm,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-zmmul,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl4096b,-zvl512b,-zvl65536b,-zvl8192b" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(read) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(write) }
attributes #6 = { noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+m,+zepi,+zicsr,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl32b,+zvl64b,-e,-experimental-smaia,-experimental-ssaia,-experimental-zca,-experimental-zcb,-experimental-zcd,-experimental-zcf,-experimental-zfa,-experimental-zicond,-experimental-zihintntl,-experimental-ztso,-experimental-zvbb,-experimental-zvbc,-experimental-zvfh,-experimental-zvkg,-experimental-zvkn,-experimental-zvkned,-experimental-zvkng,-experimental-zvknha,-experimental-zvknhb,-experimental-zvks,-experimental-zvksed,-experimental-zvksg,-experimental-zvksh,-experimental-zvkt,-h,-relax,-save-restore,-svinval,-svnapot,-svpbmt,-v,-xsfvcp,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zdinx,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zicbom,-zicbop,-zicboz,-zicntr,-zifencei,-zihintpause,-zihpm,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-zmmul,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl4096b,-zvl512b,-zvl65536b,-zvl8192b" }
attributes #7 = { nounwind }
attributes #8 = { noreturn nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!38, !39, !40, !41, !42, !43, !44, !45}
!llvm.ident = !{!46}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !7, line: 26, type: !30, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "addr2line-01.c", directory: "/home/rferrer/vehave/vehave-build", checksumkind: CSK_MD5, checksum: "e96e4e21d7a41db39f4766c1fa0cd0ca")
!4 = !{!5, !12, !17, !22, !0, !28, !34, !36}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(scope: null, file: !7, line: 49, type: !8, isLocal: true, isDefinition: true)
!7 = !DIFile(filename: "vehave-src/tests/paraver/addr2line-01.c", directory: "/home/rferrer/vehave")
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 120, elements: !10)
!9 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!10 = !{!11}
!11 = !DISubrange(count: 15)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(scope: null, file: !7, line: 49, type: !14, isLocal: true, isDefinition: true)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 304, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: 38)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(scope: null, file: !7, line: 49, type: !19, isLocal: true, isDefinition: true)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 488, elements: !20)
!20 = !{!21}
!21 = !DISubrange(count: 61)
!22 = !DIGlobalVariableExpression(var: !23, expr: !DIExpression())
!23 = distinct !DIGlobalVariable(scope: null, file: !7, line: 49, type: !24, isLocal: true, isDefinition: true)
!24 = !DICompositeType(tag: DW_TAG_array_type, baseType: !25, size: 184, elements: !26)
!25 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!26 = !{!27}
!27 = !DISubrange(count: 23)
!28 = !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !7, line: 26, type: !30, isLocal: false, isDefinition: true)
!30 = !DICompositeType(tag: DW_TAG_array_type, baseType: !31, size: 4096, elements: !32)
!31 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!32 = !{!33}
!33 = !DISubrange(count: 64)
!34 = !DIGlobalVariableExpression(var: !35, expr: !DIExpression())
!35 = distinct !DIGlobalVariable(name: "test", scope: !2, file: !7, line: 26, type: !30, isLocal: false, isDefinition: true)
!36 = !DIGlobalVariableExpression(var: !37, expr: !DIExpression())
!37 = distinct !DIGlobalVariable(name: "ref", scope: !2, file: !7, line: 26, type: !30, isLocal: false, isDefinition: true)
!38 = !{i32 7, !"Dwarf Version", i32 5}
!39 = !{i32 2, !"Debug Info Version", i32 3}
!40 = !{i32 1, !"wchar_size", i32 4}
!41 = !{i32 1, !"target-abi", !"lp64d"}
!42 = !{i32 8, !"PIC Level", i32 2}
!43 = !{i32 7, !"PIE Level", i32 2}
!44 = !{i32 7, !"uwtable", i32 2}
!45 = !{i32 8, !"SmallDataLimit", i32 8}
!46 = !{!"clang version 17.0.0"}
!47 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 28, type: !48, scopeLine: 29, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !53)
!48 = !DISubroutineType(types: !49)
!49 = !{!50, !50, !51}
!50 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!51 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !52, size: 64)
!52 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!53 = !{!54, !55, !56, !57, !58, !62, !63}
!54 = !DILocalVariable(name: "argc", arg: 1, scope: !47, file: !7, line: 28, type: !50)
!55 = !DILocalVariable(name: "argv", arg: 2, scope: !47, file: !7, line: 28, type: !51)
!56 = !DILocalVariable(name: "gvl", scope: !47, file: !7, line: 30, type: !50)
!57 = !DILocalVariable(name: "i", scope: !47, file: !7, line: 32, type: !50)
!58 = !DILocalVariable(name: "va", scope: !47, file: !7, line: 40, type: !59)
!59 = !DICompositeType(tag: DW_TAG_array_type, baseType: !31, size: 64, flags: DIFlagVector, elements: !60)
!60 = !{!61}
!61 = !DISubrange(count: 1)
!62 = !DILocalVariable(name: "vb", scope: !47, file: !7, line: 41, type: !59)
!63 = !DILocalVariable(name: "vc", scope: !47, file: !7, line: 43, type: !59)
!64 = !{!65, !65, i64 0}
!65 = !{!"int", !66, i64 0}
!66 = !{!"omnipotent char", !67, i64 0}
!67 = !{!"Simple C/C++ TBAA"}
!68 = !DILocation(line: 28, column: 14, scope: !47)
!69 = !{!70, !70, i64 0}
!70 = !{!"any pointer", !66, i64 0}
!71 = !DILocation(line: 28, column: 26, scope: !47)
!72 = !DILocation(line: 30, column: 3, scope: !47)
!73 = !DILocation(line: 30, column: 7, scope: !47)
!74 = !DILocation(line: 30, column: 13, scope: !47)
!75 = !DILocation(line: 32, column: 3, scope: !47)
!76 = !DILocation(line: 32, column: 7, scope: !47)
!77 = !DILocation(line: 33, column: 10, scope: !78)
!78 = distinct !DILexicalBlock(scope: !47, file: !7, line: 33, column: 3)
!79 = !DILocation(line: 33, column: 8, scope: !78)
!80 = !DILocation(line: 33, column: 15, scope: !81)
!81 = distinct !DILexicalBlock(scope: !78, file: !7, line: 33, column: 3)
!82 = !DILocation(line: 33, column: 19, scope: !81)
!83 = !DILocation(line: 33, column: 17, scope: !81)
!84 = !DILocation(line: 33, column: 3, scope: !78)
!85 = !DILocation(line: 35, column: 14, scope: !86)
!86 = distinct !DILexicalBlock(scope: !81, file: !7, line: 34, column: 3)
!87 = !DILocation(line: 35, column: 13, scope: !86)
!88 = !DILocation(line: 35, column: 12, scope: !86)
!89 = !DILocation(line: 35, column: 7, scope: !86)
!90 = !DILocation(line: 35, column: 5, scope: !86)
!91 = !DILocation(line: 35, column: 10, scope: !86)
!92 = !{!93, !93, i64 0}
!93 = !{!"long", !66, i64 0}
!94 = !DILocation(line: 36, column: 14, scope: !86)
!95 = !DILocation(line: 36, column: 13, scope: !86)
!96 = !DILocation(line: 36, column: 12, scope: !86)
!97 = !DILocation(line: 36, column: 7, scope: !86)
!98 = !DILocation(line: 36, column: 5, scope: !86)
!99 = !DILocation(line: 36, column: 10, scope: !86)
!100 = !DILocation(line: 37, column: 16, scope: !86)
!101 = !DILocation(line: 37, column: 15, scope: !86)
!102 = !DILocation(line: 37, column: 14, scope: !86)
!103 = !DILocation(line: 37, column: 9, scope: !86)
!104 = !DILocation(line: 37, column: 5, scope: !86)
!105 = !DILocation(line: 37, column: 12, scope: !86)
!106 = !DILocation(line: 38, column: 3, scope: !86)
!107 = !DILocation(line: 33, column: 25, scope: !81)
!108 = !DILocation(line: 33, column: 3, scope: !81)
!109 = distinct !{!109, !84, !110, !111}
!110 = !DILocation(line: 38, column: 3, scope: !78)
!111 = !{!"llvm.loop.mustprogress"}
!112 = !DILocation(line: 40, column: 3, scope: !47)
!113 = !DILocation(line: 40, column: 15, scope: !47)
!114 = !DILocation(line: 40, column: 49, scope: !47)
!115 = !DILocation(line: 40, column: 20, scope: !47)
!116 = !{!66, !66, i64 0}
!117 = !DILocation(line: 41, column: 3, scope: !47)
!118 = !DILocation(line: 41, column: 15, scope: !47)
!119 = !DILocation(line: 41, column: 49, scope: !47)
!120 = !DILocation(line: 41, column: 20, scope: !47)
!121 = !DILocation(line: 43, column: 3, scope: !47)
!122 = !DILocation(line: 43, column: 15, scope: !47)
!123 = !DILocation(line: 43, column: 45, scope: !47)
!124 = !DILocation(line: 43, column: 49, scope: !47)
!125 = !DILocation(line: 43, column: 53, scope: !47)
!126 = !DILocation(line: 43, column: 20, scope: !47)
!127 = !DILocation(line: 45, column: 36, scope: !47)
!128 = !DILocation(line: 45, column: 40, scope: !47)
!129 = !DILocation(line: 45, column: 3, scope: !47)
!130 = !DILocation(line: 47, column: 10, scope: !131)
!131 = distinct !DILexicalBlock(scope: !47, file: !7, line: 47, column: 3)
!132 = !DILocation(line: 47, column: 8, scope: !131)
!133 = !DILocation(line: 47, column: 15, scope: !134)
!134 = distinct !DILexicalBlock(scope: !131, file: !7, line: 47, column: 3)
!135 = !DILocation(line: 47, column: 19, scope: !134)
!136 = !DILocation(line: 47, column: 17, scope: !134)
!137 = !DILocation(line: 47, column: 3, scope: !131)
!138 = !DILocation(line: 49, column: 5, scope: !139)
!139 = distinct !DILexicalBlock(scope: !140, file: !7, line: 49, column: 5)
!140 = distinct !DILexicalBlock(scope: !141, file: !7, line: 49, column: 5)
!141 = distinct !DILexicalBlock(scope: !134, file: !7, line: 48, column: 3)
!142 = !DILocation(line: 49, column: 5, scope: !140)
!143 = !DILocation(line: 50, column: 3, scope: !141)
!144 = !DILocation(line: 47, column: 25, scope: !134)
!145 = !DILocation(line: 47, column: 3, scope: !134)
!146 = distinct !{!146, !137, !147, !111}
!147 = !DILocation(line: 50, column: 3, scope: !131)
!148 = !DILocation(line: 53, column: 1, scope: !47)
!149 = !DILocation(line: 52, column: 3, scope: !47)
!150 = !DISubprogram(name: "__assert_fail", scope: !151, file: !151, line: 67, type: !152, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized, retainedNodes: !156)
!151 = !DIFile(filename: "/usr/include/assert.h", directory: "")
!152 = !DISubroutineType(types: !153)
!153 = !{null, !154, !154, !155, !154}
!154 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64)
!155 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!156 = !{}
