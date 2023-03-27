//==- RISCVSchedAvispado.td - Avispado Scheduling Defs -------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the simpler per-operand machine model.
// This works with MachineScheduler. See MCSchedule.h for details.

// Avispado machine model for scheduling and other instruction cost heuristics.
def AvispadoModel : SchedMachineModel {
  let MicroOpBufferSize = 8; // This is a decoupled design,
                             // so the instruction buffer is large.
                             // FIXME: This is for the VPU but what about the
                             // scalar core.
  let IssueWidth = 1;        // One instruction per cycle can be dispatched.
  let LoadLatency = 3;
  let MispredictPenalty = 3;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr];

  // We should be able to remove this when all the instructions
  // have scheduling information.
  let CompleteModel = 0;
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Modeling each pipeline as a ProcResource using the BufferSize = 0 since
// Avispado is in-order.

let BufferSize = 0 in {
def AvispadoUnitALU        : ProcResource<1>; // Int ALU
def AvispadoUnitIMul       : ProcResource<1>; // Int Multiply
def AvispadoUnitMem        : ProcResource<1>; // Load/Store
def AvispadoUnitB          : ProcResource<1>; // Branch

def AvispadoUnitFPALU      : ProcResource<1>; // FP ALU
}

// FIXME: This was taken from Rocket so perhaps doesn't make sense for Avispado.
let BufferSize = 1 in {
def AvispadoUnitIDiv       : ProcResource<1>; // Int Division
def AvispadoUnitFPDivSqrt  : ProcResource<1>; // FP Divide/Sqrt
}

//===----------------------------------------------------------------------===//

let SchedModel = AvispadoModel in {

// Branching
def : WriteRes<WriteJmp, [AvispadoUnitB]>;
def : WriteRes<WriteJal, [AvispadoUnitB]>;
def : WriteRes<WriteJalr, [AvispadoUnitB]>;
def : WriteRes<WriteJmpReg, [AvispadoUnitB]>;

// Integer arithmetic and logic
def : WriteRes<WriteIALU32, [AvispadoUnitALU]>;
def : WriteRes<WriteIALU, [AvispadoUnitALU]>;
def : WriteRes<WriteShiftReg32, [AvispadoUnitALU]>;
def : WriteRes<WriteShiftImm32, [AvispadoUnitALU]>;
def : WriteRes<WriteShiftReg, [AvispadoUnitALU]>;
def : WriteRes<WriteShiftImm, [AvispadoUnitALU]>;

// Integer multiplication
let Latency = 4 in {
def : WriteRes<WriteIMul, [AvispadoUnitIMul]>;
def : WriteRes<WriteIMul32, [AvispadoUnitIMul]>;
}

// Integer division
// Worst case latency is used.
def : WriteRes<WriteIDiv32, [AvispadoUnitIDiv]> {
  let Latency = 34;
  let ResourceCycles = [34];
}
def : WriteRes<WriteIDiv, [AvispadoUnitIDiv]> {
  let Latency = 33;
  let ResourceCycles = [33];
}

// Memory
def : WriteRes<WriteSTB, [AvispadoUnitMem]>;
def : WriteRes<WriteSTH, [AvispadoUnitMem]>;
def : WriteRes<WriteSTW, [AvispadoUnitMem]>;
def : WriteRes<WriteSTD, [AvispadoUnitMem]>;
def : WriteRes<WriteFST32, [AvispadoUnitMem]>;
def : WriteRes<WriteFST64, [AvispadoUnitMem]>;

let Latency = 3 in {
def : WriteRes<WriteLDB, [AvispadoUnitMem]>;
def : WriteRes<WriteLDH, [AvispadoUnitMem]>;
}

let Latency = 2 in {
def : WriteRes<WriteLDW, [AvispadoUnitMem]>;
def : WriteRes<WriteLDD, [AvispadoUnitMem]>;
def : WriteRes<WriteFLD32, [AvispadoUnitMem]>;
def : WriteRes<WriteFLD64, [AvispadoUnitMem]>;

// Atomic memory
def : WriteRes<WriteAtomicW, [AvispadoUnitMem]>;
def : WriteRes<WriteAtomicD, [AvispadoUnitMem]>;

def : WriteRes<WriteAtomicLDW, [AvispadoUnitMem]>;
def : WriteRes<WriteAtomicLDD, [AvispadoUnitMem]>;
}

def : WriteRes<WriteAtomicSTW, [AvispadoUnitMem]>;
def : WriteRes<WriteAtomicSTD, [AvispadoUnitMem]>;

// Single precision.
let Latency = 4 in {
def : WriteRes<WriteFAdd32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFSGNJ32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMinMax32, [AvispadoUnitFPALU]>;
}

// Double precision
let Latency = 6 in {
def : WriteRes<WriteFAdd64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFSGNJ64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMinMax64, [AvispadoUnitFPALU]>;
}

// Conversions
let Latency = 2 in {
def : WriteRes<WriteFCvtI32ToF32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtI32ToF64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtI64ToF32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtI64ToF64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF32ToI32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF32ToI64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF64ToI32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF64ToI64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF32ToF64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF64ToF32, [AvispadoUnitFPALU]>;

def : WriteRes<WriteFClass32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFClass64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCmp32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCmp64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMovF32ToI32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMovI32ToF32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMovF64ToI64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMovI64ToF64, [AvispadoUnitFPALU]>;
}

// FP multiplication
let Latency = 5 in {
def : WriteRes<WriteFMul32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMA32, [AvispadoUnitFPALU]>;
}

let Latency = 7 in {
def : WriteRes<WriteFMul64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMA64, [AvispadoUnitFPALU]>;
}

// FP division
// FP division unit on Avispado is not pipelined, so set resource cycles to latency.
let Latency = 20, ResourceCycles = [20] in {
def : WriteRes<WriteFDiv32, [AvispadoUnitFPDivSqrt]>;
def : WriteRes<WriteFDiv64, [AvispadoUnitFPDivSqrt]>;
}

// FP square root unit on Avispado is not pipelined, so set resource cycles to latency.
def : WriteRes<WriteFSqrt32, [AvispadoUnitFPDivSqrt]> { let Latency = 20;
                                                      let ResourceCycles = [20]; }
def : WriteRes<WriteFSqrt64, [AvispadoUnitFPDivSqrt]> { let Latency = 25;
                                                      let ResourceCycles = [25]; }

// Others
def : WriteRes<WriteCSR, []>;
def : WriteRes<WriteNop, []>;

def : InstRW<[WriteIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypass and advance
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;

}

// V-extension in Avispado
def AvispadoVSETVL          : ProcResource<1>;  // Vector configuration

let BufferSize = 8 in {
  def AvispadoVPUIArith      : ProcResource<1>; // Arithmetic VPU Integer ALU
  def AvispadoVPUFArith      : ProcResource<1>; // Arithmetic VPU FP ALU
  def AvispadoVPUMemLoad    : ProcResource<1>;  // VPU Load
  def AvispadoVPUMemStore   : ProcResource<1>;  // VPU Store
}

let SchedModel = AvispadoModel in {

let Latency = 1 in
{
  def : WriteRes<WriteVSETIVLI, [AvispadoVSETVL]>;
  def : WriteRes<WriteVSETVLI,  [AvispadoVSETVL]>;
  // FIXME: This one might stall the CPU so it needs better modelling.
  def : WriteRes<WriteVSETVL,  [AvispadoVSETVL]>;
}

let Latency = 16, ResourceCycles = [8] in
{
  defm "" : LMULWriteRes<"WriteVLDE", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDM", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDS8", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDS16", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDS32", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDS64", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDUX8", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDUX16", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDUX32", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDUX64", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDOX8", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDOX16", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDOX32", [AvispadoVPUMemLoad]>;
  defm "" : LMULWriteRes<"WriteVLDOX64", [AvispadoVPUMemLoad]>;

  defm "" : LMULWriteRes<"WriteVLDFF", [AvispadoVPUMemLoad]>;

  def : WriteRes<WriteVLD1R, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD2R, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD4R, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD8R, [AvispadoVPUMemLoad]>;

  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defm "" : LMULWriteRes<"WriteVLSEG" # nf # "e" # eew, [AvispadoVPUMemLoad]>;
      defm "" : LMULWriteRes<"WriteVSSEG" # nf # "e" # eew, [AvispadoVPUMemLoad]>;
      defm "" : LMULWriteRes<"WriteVLSEGFF" # nf # "e" # eew, [AvispadoVPUMemLoad]>;
      defm "" : LMULWriteRes<"WriteVLSSEG" # nf # "e" # eew, [AvispadoVPUMemLoad]>;
      defm "" : LMULWriteRes<"WriteVSSSEG" # nf # "e" # eew, [AvispadoVPUMemLoad]>;
      defm "" : LMULWriteRes<"WriteVLUXSEG" # nf # "e" # eew, [AvispadoVPUMemLoad]>;
      defm "" : LMULWriteRes<"WriteVSUXSEG" # nf # "e" # eew, [AvispadoVPUMemLoad]>;
      defm "" : LMULWriteRes<"WriteVLOXSEG" # nf # "e" # eew, [AvispadoVPUMemLoad]>;
      defm "" : LMULWriteRes<"WriteVSOXSEG" # nf # "e" # eew, [AvispadoVPUMemLoad]>;
    }
  }
}

let Latency = 16, ResourceCycles = [16] in
{
  defm "" : LMULWriteRes<"WriteVSTUX8", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTUX16", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTUX32", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTUX64", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTOX8", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTOX16", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTOX32", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTOX64", [AvispadoVPUMemStore]>;

  defm "" : LMULWriteRes<"WriteVSTM", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTE", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTS8", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTS16", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTS32", [AvispadoVPUMemStore]>;
  defm "" : LMULWriteRes<"WriteVSTS64", [AvispadoVPUMemStore]>;

  def : WriteRes<WriteVST1R, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVST2R, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVST4R, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVST8R, [AvispadoVPUMemStore]>;
}

let Latency = 8, ResourceCycles = [8] in
{
  // FIXME: Values are likely unrealistic for LMUL>1
  defm "" : LMULWriteRes<"WriteVIALUV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIALUX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIALUI", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVExtV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVICALUV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVICALUX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVICALUI", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVShiftV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVShiftX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVShiftI", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVICmpV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVICmpX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVICmpI", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIMulV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIMulX", [AvispadoVPUIArith]>;
  defm "" : LMULSEWWriteRes<"WriteVIDivV", [AvispadoVPUIArith]>;
  defm "" : LMULSEWWriteRes<"WriteVIDivX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIMulAddV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIMulAddX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIMergeV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIMergeX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIMergeI", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIMovV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIMovX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIMovI", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVSALUV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVSALUX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVSALUI", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVAALUV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVAALUX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVSMulV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVSMulX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVSShiftV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVSShiftX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVSShiftI", [AvispadoVPUIArith]>;

  defm "" : LMULWriteResW<"WriteVIWALUV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVIWALUX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVIWALUI", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVNShiftV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVNShiftX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVNShiftI", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVIWMulV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVIWMulX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVIWMulAddV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVIWMulAddX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVNClipV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVNClipX", [AvispadoVPUIArith]>;
  defm "" : LMULWriteResW<"WriteVNClipI", [AvispadoVPUIArith]>;

  defm "" : LMULWriteRes<"WriteVIRedV_From", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVIWRedV_From", [AvispadoVPUIArith]>;
  // defm "" : LMULWriteResWRed<"WriteVIWRedV_From", [AvispadoVPUIArith]>;

  defm "" : LMULWriteRes<"WriteVMALUV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVMPopV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVMFFSV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVMSFSV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVMIotV", [AvispadoVPUIArith]>;
  defm "" : LMULWriteRes<"WriteVMIdxV", [AvispadoVPUIArith]>;
}

let Latency = 10, ResourceCycles = [10] in
{
  defm "" : LMULWriteRes<"WriteVFALUV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFALUF", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFW<"WriteVFWALUV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFW<"WriteVFWALUF", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFMulV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFMulF", [AvispadoVPUFArith]>;
  defm "" : LMULSEWWriteRes<"WriteVFDivV", [AvispadoVPUFArith]>;
  defm "" : LMULSEWWriteRes<"WriteVFDivF", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFW<"WriteVFWMulV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFW<"WriteVFWMulF", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFMulAddV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFMulAddF", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFW<"WriteVFWMulAddV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFW<"WriteVFWMulAddF", [AvispadoVPUFArith]>;
  defm "" : LMULSEWWriteRes<"WriteVFSqrtV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFRecpV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFCmpV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFCmpF", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFSgnjV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFSgnjF", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFClassV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFMergeV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFMovV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFCvtIToFV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFCvtFToIV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResW<"WriteVFWCvtIToFV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFW<"WriteVFWCvtFToIV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFW<"WriteVFWCvtFToFV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFW<"WriteVFNCvtIToFV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResW<"WriteVFNCvtFToIV", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFW<"WriteVFNCvtFToFV", [AvispadoVPUFArith]>;

  defm "" : LMULWriteRes<"WriteVFRedV_From", [AvispadoVPUFArith]>;
  defm "" : LMULWriteRes<"WriteVFRedOV_From", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFWRed<"WriteVFWRedV_From", [AvispadoVPUFArith]>;
  defm "" : LMULWriteResFWRed<"WriteVFWRedOV_From", [AvispadoVPUFArith]>;
  // def : WriteRes<WriteVFRedV, [AvispadoVPUFArith]>;
  // def : WriteRes<WriteVFRedOV, [AvispadoVPUFArith]>;
  // def : WriteRes<WriteVFWRedV, [AvispadoVPUFArith]>;
  // def : WriteRes<WriteVFWRedOV, [AvispadoVPUFArith]>;
}

// FIXME
defm "" : LMULWriteRes<"WriteVIMovVX", []>;
defm "" : LMULWriteRes<"WriteVIMovXV", []>;
defm "" : LMULWriteRes<"WriteVFMovVF", []>;
defm "" : LMULWriteRes<"WriteVFMovFV", []>;
defm "" : LMULWriteRes<"WriteVISlideX", []>;
defm "" : LMULWriteRes<"WriteVISlideI", []>;
defm "" : LMULWriteRes<"WriteVISlide1X", []>;
defm "" : LMULWriteRes<"WriteVFSlide1F", []>;
defm "" : LMULSEWWriteRes<"WriteVRGatherVV", []>;
defm "" : LMULWriteRes<"WriteVRGatherVX", []>;
defm "" : LMULWriteRes<"WriteVRGatherVI", []>;
defm "" : LMULSEWWriteRes<"WriteVCompressV", []>;
def : WriteRes<WriteVMov1V, []>;
def : WriteRes<WriteVMov2V, []>;
def : WriteRes<WriteVMov4V, []>;
def : WriteRes<WriteVMov8V, []>;
def : WriteRes<WriteRdVLENB, []>;

// Bypasses

def : ReadAdvance<ReadVSETVLI, 0>;
def : ReadAdvance<ReadVSETVL, 0>;

// 7. Vector Loads and Stores
defm "" : LMULReadAdvance<"ReadVLDX", 0>;
defm "" : LMULReadAdvance<"ReadVSTX", 0>;
defm "" : LMULReadAdvance<"ReadVSTEV", 0>;
defm "" : LMULReadAdvance<"ReadVSTM", 0>;
defm "" : LMULReadAdvance<"ReadVLDSX", 0>;
defm "" : LMULReadAdvance<"ReadVSTSX", 0>;
defm "" : LMULReadAdvance<"ReadVSTS8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS64V", 0>;
defm "" : LMULReadAdvance<"ReadVLDUXV", 0>;
defm "" : LMULReadAdvance<"ReadVLDOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64V", 0>;
def : ReadAdvance<ReadVST1R, 0>;
def : ReadAdvance<ReadVST2R, 0>;
def : ReadAdvance<ReadVST4R, 0>;
def : ReadAdvance<ReadVST8R, 0>;

// 12. Vector Integer Arithmetic Instructions
defm "" : LMULReadAdvance<"ReadVICALUV", 0>;
defm "" : LMULReadAdvance<"ReadVICALUX", 0>;
defm "" : LMULReadAdvance<"ReadVExtV", 0>;
defm "" : LMULReadAdvance<"ReadVIALUV", 0>;
defm "" : LMULReadAdvance<"ReadVIALUX", 0>;
defm "" : LMULReadAdvance<"ReadVShiftV", 0>;
defm "" : LMULReadAdvance<"ReadVShiftX", 0>;
defm "" : LMULReadAdvance<"ReadVICmpV", 0>;
defm "" : LMULReadAdvance<"ReadVICmpX", 0>;
defm "" : LMULReadAdvance<"ReadVIMulV", 0>;
defm "" : LMULReadAdvance<"ReadVIMulX", 0>;
defm "" : LMULSEWReadAdvance<"ReadVIDivV", 0>;
defm "" : LMULSEWReadAdvance<"ReadVIDivX", 0>;
defm "" : LMULReadAdvance<"ReadVIMulAddV", 0>;
defm "" : LMULReadAdvance<"ReadVIMulAddX", 0>;
defm "" : LMULReadAdvance<"ReadVIMergeV", 0>;
defm "" : LMULReadAdvance<"ReadVIMergeX", 0>;
defm "" : LMULReadAdvance<"ReadVIMovV", 0>;
defm "" : LMULReadAdvance<"ReadVIMovX", 0>;
defm "" : LMULReadAdvance<"ReadVSALUV", 0>;
defm "" : LMULReadAdvance<"ReadVSALUX", 0>;
defm "" : LMULReadAdvance<"ReadVAALUV", 0>;
defm "" : LMULReadAdvance<"ReadVAALUX", 0>;
defm "" : LMULReadAdvance<"ReadVSMulV", 0>;
defm "" : LMULReadAdvance<"ReadVSMulX", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftV", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftX", 0>;

defm "" : LMULReadAdvanceW<"ReadVIWALUV", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWALUX", 0>;
defm "" : LMULReadAdvanceW<"ReadVNShiftV", 0>;
defm "" : LMULReadAdvanceW<"ReadVNShiftX", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulV", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulX", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulAddV", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulAddX", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipV", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipX", 0>;

// 13. Vector Fixed-Point Arithmetic Instructions

// 14. Vector Floating-Point Instructions
defm "" : LMULReadAdvance<"ReadVFALUV", 0>;
defm "" : LMULReadAdvance<"ReadVFALUF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulF", 0>;
defm "" : LMULSEWReadAdvance<"ReadVFDivV", 0>;
defm "" : LMULSEWReadAdvance<"ReadVFDivF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddF", 0>;
defm "" : LMULSEWReadAdvance<"ReadVFSqrtV", 0>;
defm "" : LMULReadAdvance<"ReadVFRecpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpF", 0>;
defm "" : LMULReadAdvance<"ReadVFSgnjV", 0>;
defm "" : LMULReadAdvance<"ReadVFSgnjF", 0>;
defm "" : LMULReadAdvance<"ReadVFClassV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeF", 0>;
defm "" : LMULReadAdvance<"ReadVFMovF", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtIToFV", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtFToIV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFWCvtIToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToIV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtIToFV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFNCvtFToIV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtFToFV", 0>;

// 15. Vector Reduction Operations
def : ReadAdvance<ReadVIRedV, 0>;
def : ReadAdvance<ReadVIRedV0, 0>;
def : ReadAdvance<ReadVIWRedV, 0>;
def : ReadAdvance<ReadVIWRedV0, 0>;
def : ReadAdvance<ReadVFRedV, 0>;
def : ReadAdvance<ReadVFRedV0, 0>;
def : ReadAdvance<ReadVFRedOV, 0>;
def : ReadAdvance<ReadVFRedOV0, 0>;
def : ReadAdvance<ReadVFWRedV, 0>;
def : ReadAdvance<ReadVFWRedV0, 0>;
def : ReadAdvance<ReadVFWRedOV, 0>;
def : ReadAdvance<ReadVFWRedOV0, 0>;

// 16. Vector Mask Instructions
defm "" : LMULReadAdvance<"ReadVMALUV", 0>;
defm "" : LMULReadAdvance<"ReadVMPopV", 0>;
defm "" : LMULReadAdvance<"ReadVMFFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMSFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMIotV", 0>;

// 17. Vector Permutation Instructions
defm "" : LMULReadAdvance<"ReadVIMovVX", 0>;
defm "" : LMULReadAdvance<"ReadVIMovXV", 0>;
defm "" : LMULReadAdvance<"ReadVIMovXX", 0>;
defm "" : LMULReadAdvance<"ReadVFMovVF", 0>;
defm "" : LMULReadAdvance<"ReadVFMovFV", 0>;
defm "" : LMULReadAdvance<"ReadVFMovFX", 0>;
defm "" : LMULReadAdvance<"ReadVISlideV", 0>;
defm "" : LMULReadAdvance<"ReadVISlideX", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideV", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideF", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_data", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVI_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVCompressV", 0>;
def : ReadAdvance<ReadVMov1V, 0>;
def : ReadAdvance<ReadVMov2V, 0>;
def : ReadAdvance<ReadVMov4V, 0>;
def : ReadAdvance<ReadVMov8V, 0>;

// Others
def : ReadAdvance<ReadVMask, 0>;

// Unsupported extensions
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfh;
defm : UnsupportedSchedSFB;
}
