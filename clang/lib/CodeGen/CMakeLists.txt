set(LLVM_LINK_COMPONENTS
  AggressiveInstCombine
  Analysis
  BitReader
  BitWriter
  CodeGenTypes
  Core
  Coroutines
  Coverage
  Demangle
  Extensions
  FrontendHLSL
  FrontendOpenMP
  IPO
  IRPrinter
  IRReader
  InstCombine
  Instrumentation
  LTO
  Linker
  MC
  ObjCARCOpts
  Object
  Passes
  ProfileData
  ScalarOpts
  Support
  Target
  TargetParser
  TransformUtils
  )

add_clang_library(clangCodeGen
  ABIInfo.cpp
  ABIInfoImpl.cpp
  BackendUtil.cpp
  CGAtomic.cpp
  CGBlocks.cpp
  CGBuiltin.cpp
  CGCUDANV.cpp
  CGCUDARuntime.cpp
  CGCXX.cpp
  CGCXXABI.cpp
  CGCall.cpp
  CGClass.cpp
  CGCleanup.cpp
  CGCoroutine.cpp
  CGDebugInfo.cpp
  CGDecl.cpp
  CGDeclCXX.cpp
  CGException.cpp
  CGExpr.cpp
  CGExprAgg.cpp
  CGExprCXX.cpp
  CGExprComplex.cpp
  CGExprConstant.cpp
  CGExprScalar.cpp
  CGGPUBuiltin.cpp
  CGHLSLRuntime.cpp
  CGLoopInfo.cpp
  CGNonTrivialStruct.cpp
  CGObjC.cpp
  CGObjCGNU.cpp
  CGObjCMac.cpp
  CGObjCRuntime.cpp
  CGOpenCLRuntime.cpp
  CGOpenMPRuntime.cpp
  CGOpenMPRuntimeGPU.cpp
  CGRecordLayoutBuilder.cpp
  CGStmt.cpp
  CGStmtOpenMP.cpp
  CGVTT.cpp
  CGVTables.cpp
  CodeGenABITypes.cpp
  CodeGenAction.cpp
  CodeGenFunction.cpp
  CodeGenModule.cpp
  CodeGenPGO.cpp
  CodeGenTBAA.cpp
  CodeGenTypes.cpp
  ConstantInitBuilder.cpp
  CoverageMappingGen.cpp
  ItaniumCXXABI.cpp
  MacroPPCallbacks.cpp
  MicrosoftCXXABI.cpp
  ModuleBuilder.cpp
  ObjectFilePCHContainerOperations.cpp
  PatternInit.cpp
  SanitizerMetadata.cpp
  SwiftCallingConv.cpp
  TargetInfo.cpp
  Targets/AArch64.cpp
  Targets/AMDGPU.cpp
  Targets/ARC.cpp
  Targets/ARM.cpp
  Targets/AVR.cpp
  Targets/BPF.cpp
  Targets/CSKY.cpp
  Targets/Hexagon.cpp
  Targets/Lanai.cpp
  Targets/LoongArch.cpp
  Targets/M68k.cpp
  Targets/MSP430.cpp
  Targets/Mips.cpp
  Targets/NVPTX.cpp
  Targets/PNaCl.cpp
  Targets/PPC.cpp
  Targets/RISCV.cpp
  Targets/SPIR.cpp
  Targets/Sparc.cpp
  Targets/SystemZ.cpp
  Targets/TCE.cpp
  Targets/VE.cpp
  Targets/WebAssembly.cpp
  Targets/X86.cpp
  Targets/XCore.cpp
  VarBypassDetector.cpp

  DEPENDS
  intrinsics_gen
  builtins_codegen_epi

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  )

# EPI
# FIXME: Do this better
if( ${CLANG_TABLEGEN} STREQUAL "clang-tblgen" )
  set(REAL_CLANG_TABLEGEN $<TARGET_FILE:clang-tblgen>)
else()
  set(REAL_CLANG_TABLEGEN ${CLANG_TABLEGEN_EXE})
endif()

add_custom_command(
  OUTPUT EPIBuiltinCodegen.inc
  COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/EPI/generate-epi-builtins-def.py
    --mode codegen
    --tablegen ${REAL_CLANG_TABLEGEN}
    --output-file EPIBuiltinCodegen.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/clang/Basic/epi_builtins.td
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/clang/Basic/epi_builtins.td
     ${CLANG_TABLEGEN_EXE}
     ${CLANG_TABLEGEN_TARGET}
     ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/EPI/generate-epi-builtins-def.py
   VERBATIM)
add_custom_target(builtins_codegen_epi ALL DEPENDS EPIBuiltinCodegen.inc)
