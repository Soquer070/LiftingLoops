//===- IntrinsicsRISCV.td - Defines EPI RISCV intrinsics ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the EPI RISCV-specific intrinsics.
//
//===----------------------------------------------------------------------===//

def llvm_epi_i8_ty : LLVMType<nxv1i8>;
def llvm_epi_i16_ty : LLVMType<nxv1i16>;
def llvm_epi_i32_ty : LLVMType<nxv1i32>;
def llvm_epi_i64_ty : LLVMType<nxv1i64>;

def llvm_epi_f32_ty : LLVMType<nxv1f32>;
def llvm_epi_f64_ty : LLVMType<nxv1f64>;

class EPIIntrinsicClassID;

def EPICIDBinary : EPIIntrinsicClassID;
def EPICIDBinaryMask : EPIIntrinsicClassID;
def EPICIDBinaryMaskIn : EPIIntrinsicClassID;
def EPICIDBinaryMaskInOut : EPIIntrinsicClassID;
def EPICIDBinaryRelational : EPIIntrinsicClassID;
def EPICIDBinaryRelationalMask : EPIIntrinsicClassID;
def EPICIDBinaryWide : EPIIntrinsicClassID;
def EPICIDBinaryWideMask : EPIIntrinsicClassID;
def EPICIDConversion : EPIIntrinsicClassID;
def EPICIDConversionMask : EPIIntrinsicClassID;
def EPICIDMaskOpUnary : EPIIntrinsicClassID;
def EPICIDMaskOpUnaryMask : EPIIntrinsicClassID;
def EPICIDTernary : EPIIntrinsicClassID;
def EPICIDTernaryMask : EPIIntrinsicClassID;
def EPICIDTernaryWide : EPIIntrinsicClassID;
def EPICIDTernaryWideMask : EPIIntrinsicClassID;
def EPICIDUnary : EPIIntrinsicClassID;
def EPICIDUnaryMask : EPIIntrinsicClassID;
def EPICIDUnaryScalar : EPIIntrinsicClassID;
def EPICIDUnaryScalarMask : EPIIntrinsicClassID;
def EPICIDReduction : EPIIntrinsicClassID;
def EPICIDReductionMask : EPIIntrinsicClassID;
def EPICIDReductionWide : EPIIntrinsicClassID;
def EPICIDReductionWideMask : EPIIntrinsicClassID;
def EPICIDvfmv_f_s : EPIIntrinsicClassID;
def EPICIDvfmv_s_f : EPIIntrinsicClassID;
def EPICIDvid : EPIIntrinsicClassID;
def EPICIDvid_mask : EPIIntrinsicClassID;
def EPICIDvmiota : EPIIntrinsicClassID;
def EPICIDvmiota_mask : EPIIntrinsicClassID;
def EPICIDvmv_x_s : EPIIntrinsicClassID;
def EPICIDvmv_s_x : EPIIntrinsicClassID;

def EPICIDload : EPIIntrinsicClassID;
def EPICIDstore : EPIIntrinsicClassID;

class EPIIntrinsic<EPIIntrinsicClassID CID> {
  // These intrinsics may accept illegal integer values in their llvm_any_ty
  // operand, so they have to be extended. If set to zero then the intrinsic
  // does not have any operand that must be extended.
  Intrinsic IntrinsicID = !cast<Intrinsic>(NAME);
  EPIIntrinsicClassID ClassID = CID;
  bits<4> ExtendOperand = 0;
  bits<4> MaskOperand = 0;
  bits<4> GVLOperand = 0;
}

let TargetPrefix = "epi" in {

def int_epi_vsetvl    : DefaultAttrsIntrinsic<[llvm_i64_ty], [
    /* requested vector length */  llvm_i64_ty,
    /* single element width */     llvm_i64_ty,
    /* vector length multiplier */ llvm_i64_ty], [ IntrNoMem ]>;
def int_epi_vsetvlmax : DefaultAttrsIntrinsic<[llvm_i64_ty], [
    /* single element width */     llvm_i64_ty,
    /* vector length multiplier */ llvm_i64_ty], [ IntrNoMem ]>;

class EPIBinaryCommon<EPIIntrinsicClassID EPICID>
      : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                   [ LLVMMatchType<0>, llvm_any_ty, llvm_i64_ty ],
                   [ IntrNoMem ] >, EPIIntrinsic<EPICID>
{
  let ExtendOperand = 2;
  let GVLOperand = 3;
}
class EPIBinaryCommonMask<EPIIntrinsicClassID EPICID>
     : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                  [ LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                    llvm_anyvector_ty, llvm_i64_ty ],
                  [ IntrNoMem ] >, EPIIntrinsic<EPICID>
{
  let ExtendOperand = 3;
  let MaskOperand = 4;
  let GVLOperand = 5;
}
multiclass epi_binary {
  def "int_epi_" # NAME : EPIBinaryCommon<EPICIDBinary>;
  def "int_epi_" # NAME # "_mask" : EPIBinaryCommonMask<EPICIDBinaryMask>;
}
multiclass epi_binary_nomask {
  def "int_epi_" # NAME : EPIBinaryCommon<EPICIDBinary>;
}

multiclass epi_reduction {
  def "int_epi_" # NAME : EPIBinaryCommon<EPICIDReduction>;
  def "int_epi_" # NAME # "_mask" : EPIBinaryCommonMask<EPICIDReductionMask>;
}

class EPIBinaryMaskIn : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                                   [ LLVMMatchType<0>, llvm_any_ty,
                                     llvm_anyvector_ty, llvm_i64_ty ],
                                   [ IntrNoMem ] >,
                        EPIIntrinsic<EPICIDBinaryMaskIn>
{
  let ExtendOperand = 2;
  let MaskOperand = 3;
  let GVLOperand = 4;
}
multiclass epi_binary_mask_in {
  def "int_epi_" # NAME : EPIBinaryMaskIn;
}

class EPIBinaryMaskInOut : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                                      [ llvm_anyvector_ty, llvm_any_ty,
                                        LLVMMatchType<0>, llvm_i64_ty ],
                                      [ IntrNoMem ] >,
                           EPIIntrinsic<EPICIDBinaryMaskInOut>
{
  let ExtendOperand = 2;
  let MaskOperand = 3;
  let GVLOperand = 4;
}
multiclass epi_binary_mask_in_out {
  def "int_epi_" # NAME : EPIBinaryMaskInOut;
}

class EPIBinaryWideCommon<EPIIntrinsicClassID EPICID>
      : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                   [ llvm_anyvector_ty, llvm_any_ty, llvm_i64_ty ],
                   [ IntrNoMem ] >, EPIIntrinsic<EPICID>
{
  let ExtendOperand = 2;
  let GVLOperand = 3;
}
class EPIBinaryWideCommonMask<EPIIntrinsicClassID EPICID>
      : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                   [ LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     llvm_anyvector_ty, llvm_i64_ty ],
                   [ IntrNoMem ] >, EPIIntrinsic<EPICID>
{
  let ExtendOperand = 3;
  let MaskOperand = 4;
  let GVLOperand = 5;
}
multiclass epi_binary_wide {
  def "int_epi_" # NAME : EPIBinaryWideCommon<EPICIDBinaryWide>;
  def "int_epi_" # NAME # "_mask" : EPIBinaryWideCommonMask<EPICIDBinaryWideMask>;
}

multiclass epi_reduction_wide {
  def "int_epi_" # NAME : EPIBinaryWideCommon<EPICIDReductionWide>;
  def "int_epi_" # NAME # "_mask" : EPIBinaryWideCommonMask<EPICIDReductionWideMask>;
}

class EPIBinaryRelational : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                      [ llvm_anyvector_ty, llvm_any_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic<EPICIDBinaryRelational>
{
  let ExtendOperand = 2;
  let GVLOperand = 3;
}
class EPIBinaryRelationalMask : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                        LLVMMatchType<0>, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic<EPICIDBinaryRelationalMask>
{
  let ExtendOperand = 3;
  let MaskOperand = 4;
  let GVLOperand = 5;
}
multiclass epi_relational {
  def "int_epi_" # NAME : EPIBinaryRelational;
  def "int_epi_" # NAME # "_mask" : EPIBinaryRelationalMask;
}
multiclass epi_relational_nomask {
  def "int_epi_" # NAME : EPIBinaryRelational;
}

class EPIUnary : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                      [ llvm_any_ty, llvm_i64_ty],
                      [ IntrNoMem ] >, EPIIntrinsic<EPICIDUnary>
{
  let ExtendOperand = 1;
  let GVLOperand = 2;
}
class EPIUnaryMask : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_any_ty, llvm_anyvector_ty,
                        llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic<EPICIDUnaryMask>
{
  let ExtendOperand = 2;
  let MaskOperand = 3;
  let GVLOperand = 4;
}
multiclass epi_unary {
  def "int_epi_" # NAME : EPIUnary;
  def "int_epi_" # NAME # "_mask" : EPIUnaryMask;
}

class EPIUnaryScalar : DefaultAttrsIntrinsic< [ llvm_i64_ty ],
                      [ llvm_anyvector_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic<EPICIDUnaryScalar>;
class EPIUnaryScalarMask : DefaultAttrsIntrinsic< [ llvm_i64_ty ],
                      [ llvm_anyvector_ty, LLVMMatchType<0>, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic<EPICIDUnaryScalarMask>;
multiclass epi_unary_mask_to_scalar {
  def "int_epi_" # NAME : EPIUnaryScalar;
  def "int_epi_" # NAME # "_mask" : EPIUnaryScalarMask;
}

class EPITernary : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                        llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic<EPICIDTernary>
{
  let ExtendOperand = 2;
  let GVLOperand = 4;
}
class EPITernaryMask : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                      [ LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                        llvm_anyvector_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic<EPICIDTernaryMask>
{
  let ExtendOperand = 2;
  let MaskOperand = 4;
  let GVLOperand = 5;
}
multiclass epi_ternary {
  def "int_epi_" # NAME : EPITernary;
  def "int_epi_" # NAME # "_mask" : EPITernaryMask;
}

class EPITernaryWide : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                      [ llvm_anyvector_ty, llvm_any_ty, LLVMMatchType<0>,
                        llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic<EPICIDTernaryWide>
{
  let ExtendOperand = 2;
}
class EPITernaryWideMask : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                      [ llvm_anyvector_ty, llvm_any_ty, LLVMMatchType<0>,
                        llvm_anyvector_ty, llvm_i64_ty ],
                      [ IntrNoMem ] >, EPIIntrinsic<EPICIDTernaryWideMask>
{
  let ExtendOperand = 2;
}
multiclass epi_ternary_wide {
  def "int_epi_" # NAME : EPITernaryWide;
  def "int_epi_" # NAME # "_mask" : EPITernaryWideMask;
}

multiclass epi_unary_mask {
def int_epi_# NAME : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                               [ LLVMMatchType<0>, llvm_i64_ty ],
                               [ IntrNoMem ]>,
                     EPIIntrinsic<EPICIDMaskOpUnary>;
def int_epi_# NAME #_mask : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                               [ LLVMMatchType<0>, LLVMMatchType<0>,
                                 llvm_i64_ty ],
                               [ IntrNoMem ]>,
                     EPIIntrinsic<EPICIDMaskOpUnaryMask>;
}

multiclass epi_conversions {
def int_epi_# NAME : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                               [ llvm_anyvector_ty, llvm_i64_ty ],
                               [ IntrNoMem ]>,
                     EPIIntrinsic<EPICIDConversion>;
def int_epi_# NAME #_mask : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                               [ LLVMMatchType<0>, llvm_anyvector_ty,
                                 llvm_anyvector_ty, llvm_i64_ty ],
                               [ IntrNoMem ]>,
                     EPIIntrinsic<EPICIDConversionMask>;
}


defm vadd : epi_binary;
defm vsub : epi_binary;
defm vrsub : epi_binary;

defm vwaddu : epi_binary_wide;
defm vwadd : epi_binary_wide;
defm vwaddu_w : epi_binary;
defm vwadd_w : epi_binary;
defm vwsubu : epi_binary_wide;
defm vwsub : epi_binary_wide;
defm vwsubu_w : epi_binary;
defm vwsub_w : epi_binary;

defm vadc : epi_binary_mask_in;
defm vmadc : epi_relational_nomask;
defm vmadc_carry_in : epi_binary_mask_in_out;

defm vsbc : epi_binary_mask_in;
defm vmsbc : epi_relational_nomask;
defm vmsbc_borrow_in : epi_binary_mask_in_out;

defm vand : epi_binary;
defm vor : epi_binary;
defm vxor : epi_binary;

defm vsll : epi_binary;
defm vsrl : epi_binary;
defm vsra : epi_binary;

// These actually do narrowing but type-wise they
// are like their widening counterparts
defm vnsrl : epi_binary_wide;
defm vnsra : epi_binary_wide;

defm vmseq : epi_relational;
defm vmsne : epi_relational;
defm vmsltu : epi_relational;
defm vmslt : epi_relational;
defm vmsleu : epi_relational;
defm vmsle : epi_relational;
defm vmsgtu : epi_relational;
defm vmsgt : epi_relational;

defm vminu : epi_binary;
defm vmin : epi_binary;
defm vmaxu : epi_binary;
defm vmax : epi_binary;

defm vmul : epi_binary;
defm vmulh : epi_binary;
defm vmulhu : epi_binary;
defm vmulhsu : epi_binary;

defm vwmul : epi_binary_wide;
defm vwmulu : epi_binary_wide;
defm vwmulsu : epi_binary_wide;

defm vmacc : epi_ternary;
defm vnmsac : epi_ternary;
defm vmadd : epi_ternary;
defm vnmsub : epi_ternary;

defm vwmaccu : epi_ternary_wide;
defm vwmacc : epi_ternary_wide;
defm vwmsacu : epi_ternary_wide;
defm vwmsac : epi_ternary_wide;

defm vdivu : epi_binary;
defm vdiv : epi_binary;
defm vremu : epi_binary;
defm vrem : epi_binary;

defm vmerge : epi_binary_mask_in;

defm vsaddu : epi_binary;
defm vsadd : epi_binary;
defm vssubu : epi_binary;
defm vssub : epi_binary;

defm vaadd : epi_binary;
defm vasub : epi_binary;

defm vsmul : epi_binary;

defm vwsmaccu : epi_ternary_wide;
defm vwsmacc : epi_ternary_wide;
defm vwsmsacu : epi_ternary_wide;
defm vwsmsac : epi_ternary_wide;

defm vssrl : epi_binary;
defm vssra : epi_binary;

defm vnclipu : epi_binary;
defm vnclip : epi_binary;

defm vfadd : epi_binary;
defm vfsub : epi_binary;
defm vfwadd : epi_binary_wide;
defm vfwsub : epi_binary_wide;
defm vfwadd_w : epi_binary;
defm vfwsub_w : epi_binary;

defm vfmul : epi_binary;
defm vfdiv : epi_binary;
defm vfrdiv : epi_binary;

defm vfwmul : epi_binary_wide;

defm vfmadd : epi_ternary;
defm vfnmadd : epi_ternary;
defm vfmsub : epi_ternary;
defm vfnmsub : epi_ternary;
defm vfmacc : epi_ternary;
defm vfnmacc : epi_ternary;
defm vfmsac : epi_ternary;
defm vfnmsac : epi_ternary;

defm vfwmacc : epi_ternary_wide;
defm vfwnmacc : epi_ternary_wide;
defm vfwmsac : epi_ternary_wide;
defm vfwnmsac : epi_ternary_wide;

defm vfmin : epi_binary;
defm vfmax : epi_binary;

defm vfsgnj : epi_binary;
defm vfsgnjn : epi_binary;
defm vfsgnjx : epi_binary;

defm vmfeq : epi_relational;
defm vmfne : epi_relational;
defm vmflt : epi_relational;
defm vmfle : epi_relational;
defm vmfgt : epi_relational;
defm vmfge : epi_relational;

defm vfmerge : epi_binary_mask_in;

defm vredsum : epi_reduction;
defm vredand : epi_reduction;
defm vredor : epi_reduction;
defm vredxor : epi_reduction;
defm vredminu : epi_reduction;
defm vredmin : epi_reduction;
defm vredmaxu : epi_reduction;
defm vredmax : epi_reduction;

defm vwredsumu : epi_reduction_wide;
defm vwredsum : epi_reduction_wide;

defm vfredsum : epi_reduction;
defm vfredosum : epi_reduction;
defm vfredmin : epi_reduction;
defm vfredmax : epi_reduction;

defm vfwredsum : epi_reduction_wide;
defm vfwredosum : epi_reduction_wide;

defm vmandnot : epi_binary_nomask;
defm vmand : epi_binary_nomask;
defm vmor : epi_binary_nomask;
defm vmxor : epi_binary_nomask;
defm vmornot : epi_binary_nomask;
defm vmnand : epi_binary_nomask;
defm vmnor : epi_binary_nomask;
defm vmxnor : epi_binary_nomask;

// FIXME EDIV intrinsics disabled
//defm vdotu : epi_binary;
//defm vdot : epi_binary;
//defm vfdot : epi_binary;

defm vcompress : epi_binary;

defm vrgather : epi_binary;

defm vpopc : epi_unary_mask_to_scalar;
defm vfirst : epi_unary_mask_to_scalar;

defm vslideup : epi_binary;
defm vslidedown : epi_binary;
defm vslide1up : epi_binary;
defm vslide1down : epi_binary;

def int_epi_vmv_x_s : DefaultAttrsIntrinsic< [ llvm_anyint_ty ],
                                 [ llvm_anyvector_ty ],
                                 [ IntrNoMem ] >, EPIIntrinsic<EPICIDvmv_x_s>;
def int_epi_vmv_s_x : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                                 [ LLVMMatchType<0>, llvm_anyint_ty,
                                   llvm_i64_ty ],
                                 [ IntrNoMem ] >, EPIIntrinsic<EPICIDvmv_s_x>
{
  let ExtendOperand = 2;
}

def int_epi_vfmv_f_s : DefaultAttrsIntrinsic< [ llvm_anyfloat_ty ],
                                  [ llvm_anyvector_ty ],
                                  [ IntrNoMem ]>, EPIIntrinsic<EPICIDvfmv_f_s>;
def int_epi_vfmv_s_f : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                                  [ LLVMMatchType<0>, llvm_anyfloat_ty,
                                    llvm_i64_ty ],
                                  [ IntrNoMem ]>, EPIIntrinsic<EPICIDvfmv_s_f>;

defm vfsqrt : epi_unary;
defm vfclass : epi_unary;

defm vmsbf : epi_unary_mask;
defm vmsof : epi_unary_mask;
defm vmsif : epi_unary_mask;

def int_epi_viota : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                              [ llvm_anyvector_ty, llvm_i64_ty ],
                              [ IntrNoMem ]>,
                     EPIIntrinsic<EPICIDvmiota>;
def int_epi_viota_mask : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                   [ LLVMMatchType<0>, llvm_anyvector_ty,
                                     LLVMMatchType<1>, llvm_i64_ty ],
                                   [ IntrNoMem ]>,
                     EPIIntrinsic<EPICIDvmiota_mask>;

def int_epi_vid : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                             [ llvm_i64_ty ], [ IntrNoMem ]>,
                  EPIIntrinsic<EPICIDvid>;
def int_epi_vid_mask : DefaultAttrsIntrinsic< [ llvm_anyvector_ty ],
                                  [ LLVMMatchType<0>, llvm_anyvector_ty,
                                    llvm_i64_ty ], [ IntrNoMem ]>,
                  EPIIntrinsic<EPICIDvid_mask>;

defm vfcvt_xu_f : epi_conversions;
defm vfcvt_x_f : epi_conversions;
defm vfcvt_f_xu : epi_conversions;
defm vfcvt_f_x : epi_conversions;

defm vfwcvt_xu_f : epi_conversions;
defm vfwcvt_x_f : epi_conversions;
defm vfwcvt_f_xu : epi_conversions;
defm vfwcvt_f_x : epi_conversions;
defm vfwcvt_f_f : epi_conversions;

defm vfncvt_xu_f : epi_conversions;
defm vfncvt_x_f : epi_conversions;
defm vfncvt_f_xu : epi_conversions;
defm vfncvt_f_x : epi_conversions;
defm vfncvt_f_f : epi_conversions;

defm vwcvt_x_x : epi_conversions;
defm vwcvtu_x_x : epi_conversions;

defm vncvt_x_x : epi_conversions;

def int_epi_vmv_v_x : EPIUnary;
def int_epi_vfmv_v_f : EPIUnary;

// Loads
multiclass epi_load {
  let Properties = [SDNPMemOperand] in {
  def ""              : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ llvm_ptr_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_strided"      : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ llvm_ptr_ty,
                                    llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_indexed"      : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_mask"         : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ LLVMMatchType<0>,
                                    llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_strided_mask" : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ LLVMMatchType<0>,
                                    llvm_ptr_ty,
                                    llvm_i64_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_indexed_mask" : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ LLVMMatchType<0>,
                                    llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;

  def "_nt"           : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ llvm_ptr_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_nt_strided"   : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ llvm_ptr_ty,
                                    llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_nt_indexed"   : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_nt_mask"      : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ LLVMMatchType<0>,
                                    llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_nt_strided_mask" : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                     [ LLVMMatchType<0>,
                                       llvm_ptr_ty,
                                       llvm_i64_ty, llvm_anyvector_ty,
                                       llvm_i64_ty ],
                                     [ NoCapture<ArgIndex<1>>, IntrReadMem ]>,
                                     EPIIntrinsic<EPICIDload>;
  def "_nt_indexed_mask" : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                     [ LLVMMatchType<0>,
                                       llvm_ptr_ty,
                                       llvm_anyvector_ty, llvm_anyvector_ty,
                                       llvm_i64_ty ],
                                     [ NoCapture<ArgIndex<1>>, IntrReadMem ]>,
                                     EPIIntrinsic<EPICIDload>;

  def "_ext"           : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ llvm_ptr_ty,
                                    llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_ext_strided"   : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ llvm_ptr_ty,
                                    llvm_i64_ty, llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_ext_indexed"   : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_ext_mask"      : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                  [ LLVMMatchType<0>,
                                    llvm_ptr_ty,
                                    llvm_i64_ty, llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrReadMem ]>,
                                  EPIIntrinsic<EPICIDload>;
  def "_ext_strided_mask" : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                     [ LLVMMatchType<0>,
                                       llvm_ptr_ty,
                                       llvm_i64_ty, llvm_i64_ty,
                                       llvm_anyvector_ty, llvm_i64_ty ],
                                     [ NoCapture<ArgIndex<1>>, IntrReadMem ]>,
                                     EPIIntrinsic<EPICIDload>;
  def "_ext_indexed_mask" : DefaultAttrsIntrinsic<[ llvm_anyvector_ty ],
                                     [ LLVMMatchType<0>,
                                       llvm_ptr_ty,
                                       llvm_anyvector_ty, llvm_i64_ty,
                                       llvm_anyvector_ty, llvm_i64_ty ],
                                     [ NoCapture<ArgIndex<1>>, IntrReadMem ]>,
                                     EPIIntrinsic<EPICIDload>;

  } // Properties = [SDNPMemOperand]
}

defm int_epi_vload : epi_load;

foreach i = 2 ... 8 in {
  def int_epi_vlseg#i : DefaultAttrsIntrinsic<!listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1))),
                                  [ llvm_ptr_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>;
  def int_epi_vlseg#i#_strided : DefaultAttrsIntrinsic<!listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1))),
                                  [ llvm_ptr_ty,
                                    llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>;
  def int_epi_vlseg#i#_indexed : DefaultAttrsIntrinsic<!listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1))),
                                  [ llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<0>>, IntrReadMem ]>;
  def int_epi_vlseg#i#_mask : DefaultAttrsIntrinsic<!listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1))),
                                  !listconcat(!listsplat(LLVMMatchType<0>, i),
                                   [ llvm_ptr_ty,
                                     llvm_anyvector_ty, llvm_i64_ty ]),
                                  [ NoCapture<ArgIndex<i>>, IntrReadMem ]>;
  def int_epi_vlseg#i#_strided_mask : DefaultAttrsIntrinsic<!listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1))),
                                  !listconcat(!listsplat(LLVMMatchType<0>, i),
                                    [ llvm_ptr_ty,
                                    llvm_i64_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ]),
                                  [ NoCapture<ArgIndex<i>>, IntrReadMem ]>;
  def int_epi_vlseg#i#_indexed_mask : DefaultAttrsIntrinsic<!listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1))),
                                  !listconcat(!listsplat(LLVMMatchType<0>, i),
                                    [ llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ]),
                                  [ NoCapture<ArgIndex<i>>, IntrReadMem ]>;
}

// Stores
multiclass epi_store {
  let Properties = [SDNPMemOperand] in {
  def ""              : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_strided"      : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_indexed"      : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_mask"         : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_strided_mask" : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_i64_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_indexed_mask" : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;

  def "_nt"           : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_nt_strided"   : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_nt_indexed"   : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_nt_mask"      : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_nt_strided_mask" : DefaultAttrsIntrinsic<[],
                                     [ llvm_anyvector_ty,
                                       llvm_ptr_ty,
                                       llvm_i64_ty, llvm_anyvector_ty,
                                       llvm_i64_ty ],
                                     [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                     EPIIntrinsic<EPICIDstore>;
  def "_nt_indexed_mask" : DefaultAttrsIntrinsic<[],
                                     [ llvm_anyvector_ty,
                                       llvm_ptr_ty,
                                       llvm_anyvector_ty, llvm_anyvector_ty,
                                       llvm_i64_ty ],
                                     [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                     EPIIntrinsic<EPICIDstore>;

  def "_ext"           : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_ext_strided"   : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_i64_ty, llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_ext_indexed"   : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_ext_mask"      : DefaultAttrsIntrinsic<[],
                                  [ llvm_anyvector_ty,
                                    llvm_ptr_ty,
                                    llvm_i64_ty, llvm_anyvector_ty, llvm_i64_ty ],
                                  [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                  EPIIntrinsic<EPICIDstore>;
  def "_ext_strided_mask" : DefaultAttrsIntrinsic<[],
                                     [ llvm_anyvector_ty,
                                       llvm_ptr_ty,
                                       llvm_i64_ty, llvm_i64_ty,
                                       llvm_anyvector_ty, llvm_i64_ty ],
                                     [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                     EPIIntrinsic<EPICIDstore>;
  def "_ext_indexed_mask" : DefaultAttrsIntrinsic<[],
                                     [ llvm_anyvector_ty,
                                       llvm_ptr_ty,
                                       llvm_anyvector_ty, llvm_i64_ty,
                                       llvm_anyvector_ty, llvm_i64_ty ],
                                     [ NoCapture<ArgIndex<1>>, IntrWriteMem ]>,
                                     EPIIntrinsic<EPICIDstore>;
  } // Properties = [SDNPMemOperand]
}

defm int_epi_vstore : epi_store;

foreach i = 2 ... 8 in {
  def int_epi_vsseg#i : DefaultAttrsIntrinsic<[],
                                  !listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1)),
                                    [ llvm_ptr_ty, llvm_i64_ty ]),
                                  [ NoCapture<ArgIndex<i>>, IntrWriteMem ]>;
  def int_epi_vsseg#i#_strided : DefaultAttrsIntrinsic<[],
                                  !listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1)),
                                    [ llvm_ptr_ty,
                                    llvm_i64_ty, llvm_i64_ty ]),
                                  [ NoCapture<ArgIndex<i>>, IntrWriteMem ]>;
  def int_epi_vsseg#i#_indexed : DefaultAttrsIntrinsic<[],
                                  !listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1)),
                                    [ llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty ]),
                                  [ NoCapture<ArgIndex<i>>, IntrWriteMem ]>;
  def int_epi_vsseg#i#_mask : DefaultAttrsIntrinsic<[],
                                  !listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1)),
                                    [ llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_i64_ty ]),
                                  [ NoCapture<ArgIndex<i>>, IntrWriteMem ]>;
  def int_epi_vsseg#i#_strided_mask : DefaultAttrsIntrinsic<[],
                                  !listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1)),
                                    [ llvm_ptr_ty,
                                    llvm_i64_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ]),
                                  [ NoCapture<ArgIndex<i>>, IntrWriteMem ]>;
  def int_epi_vsseg#i#_indexed_mask : DefaultAttrsIntrinsic<[],
                                  !listconcat([ llvm_anyvector_ty ],
                                    !listsplat(LLVMMatchType<0>, !add(i, -1)),
                                    [ llvm_ptr_ty,
                                    llvm_anyvector_ty, llvm_anyvector_ty,
                                    llvm_i64_ty ]),
                                  [ NoCapture<ArgIndex<i>>, IntrWriteMem ]>;
}

// EPI Extensions

class EPIZipCommon
     : DefaultAttrsIntrinsic< [ llvm_anyvector_ty, LLVMMatchType<0> ],
                  [ LLVMMatchType<0>, LLVMMatchType<0>, llvm_i64_ty ],
                  [ IntrNoMem ] >;

def int_epi_vzip2 : EPIZipCommon;
def int_epi_vunzip2 : EPIZipCommon;
def int_epi_vtrn : EPIZipCommon;

} // TargetPrefix = "epi"

