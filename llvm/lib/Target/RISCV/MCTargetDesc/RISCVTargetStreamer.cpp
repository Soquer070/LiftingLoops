//===-- RISCVTargetStreamer.cpp - RISCV Target Streamer Methods -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RISCV specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "RISCVTargetStreamer.h"
#include "RISCVBaseInfo.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/RISCVAttributes.h"
#include "llvm/Support/RISCVISAInfo.h"

using namespace llvm;

RISCVTargetStreamer::RISCVTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

void RISCVTargetStreamer::finish() { finishAttributeSection(); }
void RISCVTargetStreamer::reset() {}

void RISCVTargetStreamer::emitDirectiveOptionPush() {}
void RISCVTargetStreamer::emitDirectiveOptionPop() {}
void RISCVTargetStreamer::emitDirectiveOptionPIC() {}
void RISCVTargetStreamer::emitDirectiveOptionNoPIC() {}
void RISCVTargetStreamer::emitDirectiveOptionRVC() {}
void RISCVTargetStreamer::emitDirectiveOptionNoRVC() {}
void RISCVTargetStreamer::emitDirectiveOptionRelax() {}
void RISCVTargetStreamer::emitDirectiveOptionNoRelax() {}
void RISCVTargetStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {}
void RISCVTargetStreamer::emitAttribute(unsigned Attribute, unsigned Value) {}
void RISCVTargetStreamer::finishAttributeSection() {}
void RISCVTargetStreamer::emitTextAttribute(unsigned Attribute,
                                            StringRef String) {}
void RISCVTargetStreamer::emitIntTextAttribute(unsigned Attribute,
                                               unsigned IntValue,
                                               StringRef StringValue) {}
void RISCVTargetStreamer::setTargetABI(RISCVABI::ABI ABI) {
  assert(ABI != RISCVABI::ABI_Unknown && "Improperly initialized target ABI");
  TargetABI = ABI;
}


void RISCVTargetStreamer::emitTargetAttributes(const MCSubtargetInfo &STI) {
  if (STI.hasFeature(RISCV::FeatureRVE))
    report_fatal_error("Codegen not yet implemented for RVE");

  emitAttribute(RISCVAttrs::STACK_ALIGN, RISCVAttrs::ALIGN_16);

  std::vector<std::string> FeatureVector;
  auto FeatureBits = STI.getFeatureBits();
  if (FeatureBits.test(RISCV::FeatureEPI)) {
    // We are using linkers that don't support these features.
    FeatureBits.reset(RISCV::FeatureStdExtZvl32b);
    FeatureBits.reset(RISCV::FeatureStdExtZvl64b);
    FeatureBits.reset(RISCV::FeatureStdExtZvl128b);
    FeatureBits.reset(RISCV::FeatureStdExtZvl256b);
    FeatureBits.reset(RISCV::FeatureStdExtZvl512b);
    FeatureBits.reset(RISCV::FeatureStdExtZvl1024b);
    FeatureBits.reset(RISCV::FeatureStdExtZvl2048b);
    FeatureBits.reset(RISCV::FeatureStdExtZvl4096b);
    FeatureBits.reset(RISCV::FeatureStdExtZvl8192b);
    FeatureBits.reset(RISCV::FeatureStdExtZvl16384b);
    FeatureBits.reset(RISCV::FeatureStdExtZvl32768b);
    FeatureBits.reset(RISCV::FeatureStdExtZve32x);
    FeatureBits.reset(RISCV::FeatureStdExtZve32f);
    FeatureBits.reset(RISCV::FeatureStdExtZve64x);
    FeatureBits.reset(RISCV::FeatureStdExtZve64f);
    FeatureBits.reset(RISCV::FeatureStdExtZve64d);
    FeatureBits.reset(RISCV::FeatureEPI);
  }

  auto ParseResult = RISCVFeatures::parseFeatureBits(
      STI.hasFeature(RISCV::Feature64Bit), FeatureBits);
  if (!ParseResult) {
    report_fatal_error(ParseResult.takeError());
  } else {
    auto &ISAInfo = *ParseResult;
    emitTextAttribute(RISCVAttrs::ARCH, ISAInfo->toString());
  }
}

// This part is for ascii assembly output
RISCVTargetAsmStreamer::RISCVTargetAsmStreamer(MCStreamer &S,
                                               formatted_raw_ostream &OS)
    : RISCVTargetStreamer(S), OS(OS) {}

void RISCVTargetAsmStreamer::emitDirectiveOptionPush() {
  OS << "\t.option\tpush\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionPop() {
  OS << "\t.option\tpop\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionPIC() {
  OS << "\t.option\tpic\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionNoPIC() {
  OS << "\t.option\tnopic\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionRVC() {
  OS << "\t.option\trvc\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionNoRVC() {
  OS << "\t.option\tnorvc\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionRelax() {
  OS << "\t.option\trelax\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionNoRelax() {
  OS << "\t.option\tnorelax\n";
}

void RISCVTargetAsmStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {
  OS << "\t.variant_cc\t" << Symbol.getName() << "\n";
}

void RISCVTargetAsmStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  OS << "\t.attribute\t" << Attribute << ", " << Twine(Value) << "\n";
}

void RISCVTargetAsmStreamer::emitTextAttribute(unsigned Attribute,
                                               StringRef String) {
  OS << "\t.attribute\t" << Attribute << ", \"" << String << "\"\n";
}

void RISCVTargetAsmStreamer::emitIntTextAttribute(unsigned Attribute,
                                                  unsigned IntValue,
                                                  StringRef StringValue) {}

void RISCVTargetAsmStreamer::finishAttributeSection() {}
