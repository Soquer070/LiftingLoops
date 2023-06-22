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
  let MicroOpBufferSize = 0; // This is a decoupled design,
                             // so the instruction buffer is large.
                             // However, the scalar core is in-order. The VPU is
                             // handled separately by using the 2 queues as
                             // resources (see below)
  let IssueWidth = 1;        // One instruction per cycle can be dispatched.
  let LoadLatency = 3;
  let MispredictPenalty = 3;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasStdExtZfa];

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
let BufferSize = 0 in {
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


// The VPU has two independent queues: one for arithmetics, one for memory
// operations. Once dispatched to the queue, the instruction wait for being
// scheduled on the ALUs. Dependencies and scheduling is completed by the
// VPU. Hence, the MI Scheduler does not have anything to do except to check
// if there is still space in the respective queues, once a vector
// instruction is encountered. The ProcResources below represent the spaces
// available in the queues.
let BufferSize = 0 in {
  def AvispadoVPUMemQueue     : ProcResource<8>;  // Queue for arithmetics
  def AvispadoVPUArithQueue   : ProcResource<16>; // Queue for mem operations
}

let SchedModel = AvispadoModel in {

let Latency = 1 in
{
  def : WriteRes<WriteVSETIVLI, [AvispadoVSETVL]>;
  def : WriteRes<WriteVSETVLI,  [AvispadoVSETVL]>;
  // FIXME: This one might stall the CPU so it needs better modelling.
  def : WriteRes<WriteVSETVL,  [AvispadoVSETVL]>;
}

let Latency = 86, ResourceCycles = [86] in
{
  defm "" : LMULWriteRes<"WriteVLDE", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDM", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDS8", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDS16", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDS32", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDS64", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDUX8", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDUX16", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDUX32", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDUX64", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDOX8", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDOX16", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDOX32", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVLDOX64", [AvispadoVPUMemQueue]>;

  defm "" : LMULWriteRes<"WriteVLDFF", [AvispadoVPUMemQueue]>;

  def : WriteRes<WriteVLD1R, [AvispadoVPUMemQueue]>;
  def : WriteRes<WriteVLD2R, [AvispadoVPUMemQueue]>;
  def : WriteRes<WriteVLD4R, [AvispadoVPUMemQueue]>;
  def : WriteRes<WriteVLD8R, [AvispadoVPUMemQueue]>;

  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defm "" : LMULWriteRes<"WriteVLSEG" # nf # "e" # eew, [AvispadoVPUMemQueue]>;
      defm "" : LMULWriteRes<"WriteVSSEG" # nf # "e" # eew, [AvispadoVPUMemQueue]>;
      defm "" : LMULWriteRes<"WriteVLSEGFF" # nf # "e" # eew, [AvispadoVPUMemQueue]>;
      defm "" : LMULWriteRes<"WriteVLSSEG" # nf # "e" # eew, [AvispadoVPUMemQueue]>;
      defm "" : LMULWriteRes<"WriteVSSSEG" # nf # "e" # eew, [AvispadoVPUMemQueue]>;
      defm "" : LMULWriteRes<"WriteVLUXSEG" # nf # "e" # eew, [AvispadoVPUMemQueue]>;
      defm "" : LMULWriteRes<"WriteVSUXSEG" # nf # "e" # eew, [AvispadoVPUMemQueue]>;
      defm "" : LMULWriteRes<"WriteVLOXSEG" # nf # "e" # eew, [AvispadoVPUMemQueue]>;
      defm "" : LMULWriteRes<"WriteVSOXSEG" # nf # "e" # eew, [AvispadoVPUMemQueue]>;
    }
  }
}

let Latency = 86, ResourceCycles = [86] in
{
  defm "" : LMULWriteRes<"WriteVSTUX8", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTUX16", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTUX32", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTUX64", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTOX8", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTOX16", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTOX32", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTOX64", [AvispadoVPUMemQueue]>;

  defm "" : LMULWriteRes<"WriteVSTM", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTE", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTS8", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTS16", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTS32", [AvispadoVPUMemQueue]>;
  defm "" : LMULWriteRes<"WriteVSTS64", [AvispadoVPUMemQueue]>;

  def : WriteRes<WriteVST1R, [AvispadoVPUMemQueue]>;
  def : WriteRes<WriteVST2R, [AvispadoVPUMemQueue]>;
  def : WriteRes<WriteVST4R, [AvispadoVPUMemQueue]>;
  def : WriteRes<WriteVST8R, [AvispadoVPUMemQueue]>;
}

let Latency = 86, ResourceCycles = [86] in
{
  // Avispado does not have LMUL != 1
  defm "" : LMULWriteRes<"WriteVIALUV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIALUX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIALUI", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVExtV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVICALUV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVICALUX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVICALUI", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVShiftV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVShiftX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVShiftI", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVICmpV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVICmpX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVICmpI", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMinMaxV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMinMaxX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMulV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMulX", [AvispadoVPUArithQueue]>;
  defm "" : LMULSEWWriteRes<"WriteVIDivV", [AvispadoVPUArithQueue]>;
  defm "" : LMULSEWWriteRes<"WriteVIDivX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMulAddV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMulAddX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMergeV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMergeX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMergeI", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMovV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMovX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVIMovI", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVSALUV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVSALUX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVSALUI", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVAALUV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVAALUX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVSMulV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVSMulX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVSShiftV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVSShiftX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVSShiftI", [AvispadoVPUArithQueue]>;

  defm "" : LMULWriteResW<"WriteVIWALUV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVIWALUX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVIWALUI", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVNShiftV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVNShiftX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVNShiftI", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVIWMulV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVIWMulX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVIWMulAddV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVIWMulAddX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVNClipV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVNClipX", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVNClipI", [AvispadoVPUArithQueue]>;

  defm "" : LMULSEWWriteRes<"WriteVIRedV_From", [AvispadoVPUArithQueue]>;
  defm "" : LMULSEWWriteResWRed<"WriteVIWRedV_From", [AvispadoVPUArithQueue]>;

  defm "" : LMULWriteRes<"WriteVMALUV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVMPopV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVMFFSV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVMSFSV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVMIotV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVMIdxV", [AvispadoVPUArithQueue]>;
}

let Latency = 86, ResourceCycles = [86] in
{
  defm "" : LMULWriteRes<"WriteVFALUV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFALUF", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResFW<"WriteVFWALUV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResFW<"WriteVFWALUF", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFMulV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFMulF", [AvispadoVPUArithQueue]>;
  defm "" : LMULSEWWriteResF<"WriteVFDivV", [AvispadoVPUArithQueue]>;
  defm "" : LMULSEWWriteResF<"WriteVFDivF", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResFW<"WriteVFWMulV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResFW<"WriteVFWMulF", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFMulAddV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFMulAddF", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResFW<"WriteVFWMulAddV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResFW<"WriteVFWMulAddF", [AvispadoVPUArithQueue]>;
  defm "" : LMULSEWWriteResF<"WriteVFSqrtV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFRecpV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFCmpV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFCmpF", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFSgnjV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFSgnjF", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFClassV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFMergeV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFMovV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFCvtIToFV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFCvtFToIV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVFWCvtIToFV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResFW<"WriteVFWCvtFToIV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResFW<"WriteVFWCvtFToFV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResFW<"WriteVFNCvtIToFV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResW<"WriteVFNCvtFToIV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteResFW<"WriteVFNCvtFToFV", [AvispadoVPUArithQueue]>;

  defm "" : LMULSEWWriteResF<"WriteVFRedV_From", [AvispadoVPUArithQueue]>;
  defm "" : LMULSEWWriteResF<"WriteVFRedOV_From", [AvispadoVPUArithQueue]>;
  defm "" : LMULSEWWriteResF<"WriteVFRedMinMaxV_From", [AvispadoVPUArithQueue]>;
  defm "" : LMULSEWWriteResFWRed<"WriteVFWRedV_From", [AvispadoVPUArithQueue]>;
  defm "" : LMULSEWWriteResFWRed<"WriteVFWRedOV_From", [AvispadoVPUArithQueue]>;

  defm "" : LMULWriteRes<"WriteVFMinMaxV", [AvispadoVPUArithQueue]>;
  defm "" : LMULWriteRes<"WriteVFMinMaxF", [AvispadoVPUArithQueue]>;
}

// FIXME
def : WriteRes<WriteVIMovVX, []>;
def : WriteRes<WriteVIMovXV, []>;
def : WriteRes<WriteVFMovVF, []>;
def : WriteRes<WriteVFMovFV, []>;
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
// Vector instructions for the VPU are stored in the memory or arithmetic
// queue and popped from the queue, if the VPU ALU has the resources to
// complete the instruction and all its operands are available. Hence, the
// MISched must not consider dependencies among instructions. Instead, it can
// dispatch a vector disregarding the dependencies. In this case, we use
// ReadAdvance usually serving as a bypass or early out, as a means to
// schedule the vector instructions immediately. That is why the ReadAdvance
// parameter equals the latency - 1.

def : ReadAdvance<ReadVSETVLI, 0>;
def : ReadAdvance<ReadVSETVL, 0>;

// 7. Vector Loads and Stores
def : ReadAdvance<ReadVLDX, 0>;
def : ReadAdvance<ReadVSTX, 0>;
defm "" : LMULReadAdvance<"ReadVSTEV", 85>;
defm "" : LMULReadAdvance<"ReadVSTM", 85>;
def : ReadAdvance<ReadVLDSX, 0>;
def : ReadAdvance<ReadVSTSX, 0>;
defm "" : LMULReadAdvance<"ReadVSTS8V", 85>;
defm "" : LMULReadAdvance<"ReadVSTS16V", 85>;
defm "" : LMULReadAdvance<"ReadVSTS32V", 85>;
defm "" : LMULReadAdvance<"ReadVSTS64V", 85>;
defm "" : LMULReadAdvance<"ReadVLDUXV", 85>;
defm "" : LMULReadAdvance<"ReadVLDOXV", 85>;
defm "" : LMULReadAdvance<"ReadVSTUXV", 85>;
defm "" : LMULReadAdvance<"ReadVSTUX8", 85>;
defm "" : LMULReadAdvance<"ReadVSTUX16", 85>;
defm "" : LMULReadAdvance<"ReadVSTUX32", 85>;
defm "" : LMULReadAdvance<"ReadVSTUX64", 85>;
defm "" : LMULReadAdvance<"ReadVSTUX8V", 85>;
defm "" : LMULReadAdvance<"ReadVSTUX16V", 85>;
defm "" : LMULReadAdvance<"ReadVSTUX32V", 85>;
defm "" : LMULReadAdvance<"ReadVSTUX64V", 85>;
defm "" : LMULReadAdvance<"ReadVSTOX8", 85>;
defm "" : LMULReadAdvance<"ReadVSTOX16", 85>;
defm "" : LMULReadAdvance<"ReadVSTOX32", 85>;
defm "" : LMULReadAdvance<"ReadVSTOX64", 85>;
defm "" : LMULReadAdvance<"ReadVSTOXV", 85>;
defm "" : LMULReadAdvance<"ReadVSTOX8V", 85>;
defm "" : LMULReadAdvance<"ReadVSTOX16V", 85>;
defm "" : LMULReadAdvance<"ReadVSTOX32V", 85>;
defm "" : LMULReadAdvance<"ReadVSTOX64V", 85>;
def : ReadAdvance<ReadVST1R, 0>;
def : ReadAdvance<ReadVST2R, 0>;
def : ReadAdvance<ReadVST4R, 0>;
def : ReadAdvance<ReadVST8R, 0>;

// 12. Vector Integer Arithmetic Instructions
defm "" : LMULReadAdvance<"ReadVICALUV", 85>;
defm "" : LMULReadAdvance<"ReadVICALUX", 85>;
defm "" : LMULReadAdvance<"ReadVExtV", 85>;
defm "" : LMULReadAdvance<"ReadVIALUV", 85>;
defm "" : LMULReadAdvance<"ReadVIALUX", 85>;
defm "" : LMULReadAdvance<"ReadVShiftV", 85>;
defm "" : LMULReadAdvance<"ReadVShiftX", 85>;
defm "" : LMULReadAdvance<"ReadVICmpV", 85>;
defm "" : LMULReadAdvance<"ReadVICmpX", 85>;
defm "" : LMULReadAdvance<"ReadVIMinMaxV", 85>;
defm "" : LMULReadAdvance<"ReadVIMinMaxX", 85>;
defm "" : LMULReadAdvance<"ReadVIMulV", 85>;
defm "" : LMULReadAdvance<"ReadVIMulX", 85>;
defm "" : LMULSEWReadAdvance<"ReadVIDivV", 85>;
defm "" : LMULSEWReadAdvance<"ReadVIDivX", 85>;
defm "" : LMULReadAdvance<"ReadVIMulAddV", 85>;
defm "" : LMULReadAdvance<"ReadVIMulAddX", 85>;
defm "" : LMULReadAdvance<"ReadVIMergeV", 85>;
defm "" : LMULReadAdvance<"ReadVIMergeX", 85>;
defm "" : LMULReadAdvance<"ReadVIMovV", 85>;
defm "" : LMULReadAdvance<"ReadVIMovX", 85>;
defm "" : LMULReadAdvance<"ReadVSALUV", 85>;
defm "" : LMULReadAdvance<"ReadVSALUX", 85>;
defm "" : LMULReadAdvance<"ReadVAALUV", 85>;
defm "" : LMULReadAdvance<"ReadVAALUX", 85>;
defm "" : LMULReadAdvance<"ReadVSMulV", 85>;
defm "" : LMULReadAdvance<"ReadVSMulX", 85>;
defm "" : LMULReadAdvance<"ReadVSShiftV", 85>;
defm "" : LMULReadAdvance<"ReadVSShiftX", 85>;

defm "" : LMULReadAdvanceW<"ReadVIWALUV", 85>;
defm "" : LMULReadAdvanceW<"ReadVIWALUX", 85>;
defm "" : LMULReadAdvanceW<"ReadVNShiftV", 85>;
defm "" : LMULReadAdvanceW<"ReadVNShiftX", 85>;
defm "" : LMULReadAdvanceW<"ReadVIWMulV", 85>;
defm "" : LMULReadAdvanceW<"ReadVIWMulX", 85>;
defm "" : LMULReadAdvanceW<"ReadVIWMulAddV", 85>;
defm "" : LMULReadAdvanceW<"ReadVIWMulAddX", 85>;
defm "" : LMULReadAdvanceW<"ReadVNClipV", 85>;
defm "" : LMULReadAdvanceW<"ReadVNClipX", 85>;

// 13. Vector Fixed-Point Arithmetic Instructions

// 14. Vector Floating-Point Instructions
defm "" : LMULReadAdvance<"ReadVFALUV", 85>;
defm "" : LMULReadAdvance<"ReadVFALUF", 85>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUV", 85>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUF", 85>;
defm "" : LMULReadAdvance<"ReadVFMulV", 85>;
defm "" : LMULReadAdvance<"ReadVFMulF", 85>;
defm "" : LMULSEWReadAdvanceF<"ReadVFDivV", 85>;
defm "" : LMULSEWReadAdvanceF<"ReadVFDivF", 85>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulV", 85>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulF", 85>;
defm "" : LMULReadAdvance<"ReadVFMulAddV", 85>;
defm "" : LMULReadAdvance<"ReadVFMulAddF", 85>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddV", 85>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddF", 85>;
defm "" : LMULSEWReadAdvanceF<"ReadVFSqrtV", 85>;
defm "" : LMULReadAdvance<"ReadVFMinMaxV", 85>;
defm "" : LMULReadAdvance<"ReadVFMinMaxF", 85>;
defm "" : LMULReadAdvance<"ReadVFRecpV", 85>;
defm "" : LMULReadAdvance<"ReadVFCmpV", 85>;
defm "" : LMULReadAdvance<"ReadVFCmpF", 85>;
defm "" : LMULReadAdvance<"ReadVFSgnjV", 85>;
defm "" : LMULReadAdvance<"ReadVFSgnjF", 85>;
defm "" : LMULReadAdvance<"ReadVFClassV", 85>;
defm "" : LMULReadAdvance<"ReadVFMergeV", 85>;
defm "" : LMULReadAdvance<"ReadVFMergeF", 85>;
defm "" : LMULReadAdvance<"ReadVFMovF", 85>;
defm "" : LMULReadAdvance<"ReadVFCvtIToFV", 85>;
defm "" : LMULReadAdvance<"ReadVFCvtFToIV", 85>;
defm "" : LMULReadAdvanceW<"ReadVFWCvtIToFV", 85>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToIV", 85>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToFV", 85>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtIToFV", 85>;
defm "" : LMULReadAdvanceW<"ReadVFNCvtFToIV", 85>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtFToFV", 85>;

// 15. Vector Reduction Operations
def : ReadAdvance<ReadVIRedV, 85>;
def : ReadAdvance<ReadVIRedV0, 85>;
def : ReadAdvance<ReadVIWRedV, 85>;
def : ReadAdvance<ReadVIWRedV0, 85>;
def : ReadAdvance<ReadVFRedV, 85>;
def : ReadAdvance<ReadVFRedV0, 85>;
def : ReadAdvance<ReadVFRedOV, 85>;
def : ReadAdvance<ReadVFRedOV0, 85>;
def : ReadAdvance<ReadVFWRedV, 85>;
def : ReadAdvance<ReadVFWRedV0, 85>;
def : ReadAdvance<ReadVFWRedOV, 85>;
def : ReadAdvance<ReadVFWRedOV0, 85>;

// 16. Vector Mask Instructions
defm "" : LMULReadAdvance<"ReadVMALUV", 85>;
defm "" : LMULReadAdvance<"ReadVMPopV", 85>;
defm "" : LMULReadAdvance<"ReadVMFFSV", 85>;
defm "" : LMULReadAdvance<"ReadVMSFSV", 85>;
defm "" : LMULReadAdvance<"ReadVMIotV", 85>;

// 17. Vector Permutation Instructions
def : ReadAdvance<ReadVIMovVX, 85>;
def : ReadAdvance<ReadVIMovXV, 85>;
def : ReadAdvance<ReadVIMovXX, 85>;
def : ReadAdvance<ReadVFMovVF, 85>;
def : ReadAdvance<ReadVFMovFV, 85>;
def : ReadAdvance<ReadVFMovFX, 85>;
defm "" : LMULReadAdvance<"ReadVISlideV", 85>;
defm "" : LMULReadAdvance<"ReadVISlideX", 85>;
defm "" : LMULReadAdvance<"ReadVFSlideV", 85>;
defm "" : LMULReadAdvance<"ReadVFSlideF", 85>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_data", 85>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_index", 85>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_data", 85>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_index", 85>;
defm "" : LMULReadAdvance<"ReadVRGatherVI_data", 85>;
defm "" : LMULSEWReadAdvance<"ReadVCompressV", 85>;
def : ReadAdvance<ReadVMov1V, 85>;
def : ReadAdvance<ReadVMov2V, 85>;
def : ReadAdvance<ReadVMov4V, 85>;
def : ReadAdvance<ReadVMov8V, 85>;

// Others
def : ReadAdvance<ReadVMask, 85>;

// Unsupported extensions
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfh;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedSFB;
}
