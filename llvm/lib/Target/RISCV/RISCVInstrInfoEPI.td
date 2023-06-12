//=- RISCVInstrInfoV.td - Zeou-Extension RISCV instructions -*- tblgen-*----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target Specific DAG nodes
//===----------------------------------------------------------------------===//

def riscv_shuffle_extend : SDNode<"RISCVISD::SHUFFLE_EXTEND",
                                     SDTypeProfile<1, 2,
                                      [SDTCisSameAs<0, 1>,
                                      SDTCisVec<0>, SDTCisInt<0>,
                                      SDTCisInt<2>]>>;

def riscv_sign_extend_inreg : SDNode<"RISCVISD::SIGN_EXTEND_BITS_INREG",
                                     SDTypeProfile<1, 2,
                                      [SDTCisSameAs<0, 1>,
                                      SDTCisVec<0>, SDTCisInt<0>,
                                      SDTCisInt<2>]>>;

def riscv_zero_extend_inreg : SDNode<"RISCVISD::ZERO_EXTEND_BITS_INREG",
                                     SDTypeProfile<1, 2,
                                      [SDTCisSameAs<0, 1>,
                                      SDTCisVec<0>, SDTCisInt<0>,
                                      SDTCisInt<2>]>>;

foreach tsize = 2 ... 8 in {

def riscv_vlseg#tsize : SDNode<"RISCVISD::VLSEG" # tsize,
                          SDTypeProfile<1, 3, [SDTCisPtrTy<1>, SDTCisInt<2>,
                                               SDTCisInt<3>]>,
                          [SDNPHasChain, SDNPMayLoad]>;
def riscv_vsseg#tsize : SDNode<"RISCVISD::VSSEG" # tsize,
                          SDTypeProfile<0, 4, [SDTCisPtrTy<1>, SDTCisInt<2>,
                                               SDTCisInt<3>]>,
                          [SDNPHasChain, SDNPMayStore]>;

def riscv_vlsseg#tsize : SDNode<"RISCVISD::VLSSEG" # tsize,
                          SDTypeProfile<1, 4, [SDTCisPtrTy<1>, SDTCisInt<2>,
                                               SDTCisInt<3>, SDTCisInt<4>]>,
                          [SDNPHasChain, SDNPMayLoad]>;
def riscv_vssseg#tsize : SDNode<"RISCVISD::VSSSEG" # tsize,
                          SDTypeProfile<0, 5, [SDTCisPtrTy<1>, SDTCisInt<2>,
                                               SDTCisInt<3>, SDTCisInt<4>]>,
                          [SDNPHasChain, SDNPMayStore]>;

def riscv_vlxseg#tsize : SDNode<"RISCVISD::VLXSEG" # tsize,
                          SDTypeProfile<1, 4, [SDTCisPtrTy<1>, SDTCisVec<2>,
                                               SDTCisInt<3>, SDTCisInt<4>]>,
                          [SDNPHasChain, SDNPMayLoad]>;
def riscv_vsxseg#tsize : SDNode<"RISCVISD::VSXSEG" # tsize,
                          SDTypeProfile<0, 5, [SDTCisPtrTy<1>, SDTCisVec<2>,
                                               SDTCisInt<3>, SDTCisInt<4>]>,
                          [SDNPHasChain, SDNPMayStore]>;

} // tsize


def riscv_vzip2 : SDNode<"RISCVISD::VZIP2",
                         SDTypeProfile<1, 3, [SDTCisSameAs<1, 2>,
                                             SDTCisVec<1>, SDTCisInt<3>]>>;

def riscv_vunzip2 : SDNode<"RISCVISD::VUNZIP2",
                           SDTypeProfile<1, 3, [SDTCisSameAs<1, 2>,
                                               SDTCisVec<1>, SDTCisInt<3>]>>;

def riscv_vtrn : SDNode<"RISCVISD::VTRN",
                        SDTypeProfile<1, 3, [SDTCisSameAs<1, 2>, SDTCisVec<1>,
                                            SDTCisInt<3>]>>;

// Vector reduction nodes.
def vecreduce_and : SDNode<"ISD::VECREDUCE_AND", SDTVecReduce>;
def vecreduce_or  : SDNode<"ISD::VECREDUCE_OR", SDTVecReduce>;
def vecreduce_xor : SDNode<"ISD::VECREDUCE_XOR", SDTVecReduce>;

// Vector fp reduction.
def SDTVecReduceFP : SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVec<1>,
                                          SDTCisFP<1>]>;

//===----------------------------------------------------------------------===//
// Additional definitions complementing those of RISCVInstrInfoVPseudos.td
//===----------------------------------------------------------------------===//

// Used to iterate over all non fractional LMULs.
def MxListNonFractional {
  list<LMULInfo> m = [V_M1, V_M2, V_M4, V_M8];
}

class Iota<int N>
{
  list<int> Value = !if(!eq(N, 0),
                        []<int>,
                        !listconcat(Iota<!add(N, -1)>.Value, [!add(N, -1)]));
}

//===----------------------------------------------------------------------===//
// EPI custom instructions
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// op vd, vs2, vs1, vm
class VCustom_VV_VV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVCustomInstVV<funct6, opv, (outs VR:$vd),
                     (ins VR:$vs2, VR:$vs1, VMaskOp:$vm),
                     opcodestr, "$vd, $vs2, $vs1${vm}">;
}

// FIXME: Should we use some other predicate?
let Predicates = [HasVInstructions] in {

def VZIP2_VV : VCustom_VV_VV<0b110101, OPIVV, "vzip2.vv">;
def VUNZIP2_VV : VCustom_VV_VV<0b110110, OPIVV, "vunzip2.vv">;
def VTRN_VV : VCustom_VV_VV<0b110111, OPIVV, "vtrn.vv">;

}

//===----------------------------------------------------------------------===//

class VMask<bits<1> vtype>
{
  bits<1> Value = vtype;
}

def vmask_all_lanes : VMask<0b1>;
def vmask_only_true : VMask<0b0>;

class EPILookupIntrinsic<string basename> {
  Intrinsic I = !cast<Intrinsic>("int_epi_" # basename);
}

//===----------------------------------------------------------------------===//
// Common definitions
//===----------------------------------------------------------------------===//

let Predicates = [HasVInstructions] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0,
    Uses = [VL] in
def PseudoEPIReadVL : Pseudo<(outs GPR:$rd),
                          (ins), [], "rdvl", "$rd">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1,
     usesCustomInserter = 1 in
def PseudoEPIVSCALE : Pseudo<(outs GPR:$rd), (ins), [], "">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0,
    Uses = [VTYPE] in
def PseudoEPIReadVTYPE : Pseudo<(outs GPR:$rd),
                             (ins), [], "rdvtype", "$rd">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0 in
def PseudoEPIReadVLENB : Pseudo<(outs GPR:$rd),
                             (ins), [], "rdvlenb", "$rd">;

let hasSideEffects = 1, mayLoad = 0, mayStore = 0, Defs = [VL, VTYPE] in {
def PseudoEPIVSETVL : Pseudo<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2), []>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, usesCustomInserter = 1,
    Uses = [VL, VTYPE] in
foreach MInfo = MxList in {
def "PseudoEPIVMCLR_" # MInfo.MX : Pseudo<(outs VR:$rd), (ins AVL:$vl, ixlenimm:$sew), []>;
def "PseudoEPIVMSET_" # MInfo.MX : Pseudo<(outs VR:$rd), (ins AVL:$vl, ixlenimm:$sew), []>;
}

}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

class VectorTypeInfo<ValueType Vec, ValueType Mas, int Sew, LMULInfo MI>
{
  ValueType Vector = Vec;
  ValueType Mask = Mas;
  int SEW = Sew;
  LMULInfo MInfo = MI;
}

class GroupVectorTypeInfo<ValueType Vec, ValueType VecM1, ValueType Mas,
                          int Sew, LMULInfo MI>
    : VectorTypeInfo<Vec, Mas, Sew, MI>
{
  ValueType VectorM1 = VecM1;
}

class FloatVectorTypeInfo<ValueType Vec, ValueType Mas, int Sew, LMULInfo MI,
                          ValueType Scal, RegisterClass ScalarReg>
    : VectorTypeInfo<Vec, Mas, Sew, MI>
{
  ValueType Scalar = Scal;
  RegisterClass ScalarRegClass = ScalarReg;
}

class GroupFloatVectorTypeInfo<ValueType Vec, ValueType VecM1, ValueType Mas,
                               int Sew, LMULInfo MI, ValueType Scal,
                               RegisterClass ScalarReg>
    : FloatVectorTypeInfo<Vec, Mas, Sew, MI, Scal, ScalarReg>
{
  ValueType VectorM1 = VecM1;
}

class VectorTypeInfoToWide<VectorTypeInfo vti, VectorTypeInfo wti>
{
  VectorTypeInfo Vti = vti;
  VectorTypeInfo Wti = wti;
}

class FloatVectorTypeInfoToWide<FloatVectorTypeInfo fvti, FloatVectorTypeInfo fwti>
{
  FloatVectorTypeInfo FVti = fvti;
  FloatVectorTypeInfo FWti = fwti;
}

defset list<VectorTypeInfo> EPIAllVectors = {

  defset list<VectorTypeInfo> EPIAllIntegerVectors = {
    defset list<VectorTypeInfo> EPINoGroupIntegerVectors = {
      def Vtype1xi8  : VectorTypeInfo<nxv1i8,  nxv1i1, 8,  V_MF8>;
      def Vtype2xi8  : VectorTypeInfo<nxv2i8,  nxv2i1, 8,  V_MF4>;
      def Vtype4xi8  : VectorTypeInfo<nxv4i8,  nxv4i1, 8,  V_MF2>;
      def Vtype8xi8  : VectorTypeInfo<nxv8i8,  nxv8i1, 8,  V_M1>;

      def Vtype1xi16 : VectorTypeInfo<nxv1i16, nxv1i1, 16, V_MF4>;
      def Vtype2xi16 : VectorTypeInfo<nxv2i16, nxv2i1, 16, V_MF2>;
      def Vtype4xi16 : VectorTypeInfo<nxv4i16, nxv4i1, 16, V_M1>;

      def Vtype1xi32 : VectorTypeInfo<nxv1i32, nxv1i1, 32, V_MF2>;
      def Vtype2xi32 : VectorTypeInfo<nxv2i32, nxv2i1, 32, V_M1>;

      def Vtype1xi64 : VectorTypeInfo<nxv1i64, nxv1i1, 64, V_M1>;
    }

    defset list<GroupVectorTypeInfo> EPIGroupIntegerVectors = {
      def Vtype16xi8  : GroupVectorTypeInfo<nxv16i8,  nxv8i8,  nxv16i1, 8,  V_M2>;
      def Vtype32xi8  : GroupVectorTypeInfo<nxv32i8,  nxv8i8,  nxv32i1, 8,  V_M4>;
      def Vtype64xi8  : GroupVectorTypeInfo<nxv64i8,  nxv8i8,  nxv64i1, 8,  V_M8>;

      def Vtype8xi16  : GroupVectorTypeInfo<nxv8i16,  nxv4i16, nxv8i1,  16, V_M2>;
      def Vtype16xi16 : GroupVectorTypeInfo<nxv16i16, nxv4i16, nxv16i1, 16, V_M4>;
      def Vtype32xi16 : GroupVectorTypeInfo<nxv32i16, nxv4i16, nxv32i1, 16, V_M8>;

      def Vtype4xi32  : GroupVectorTypeInfo<nxv4i32,  nxv2i32, nxv4i1,  32, V_M2>;
      def Vtype8xi32  : GroupVectorTypeInfo<nxv8i32,  nxv2i32, nxv8i1,  32, V_M4>;
      def Vtype16xi32 : GroupVectorTypeInfo<nxv16i32, nxv2i32, nxv16i1, 32, V_M8>;

      def Vtype2xi64  : GroupVectorTypeInfo<nxv2i64,  nxv1i64, nxv2i1,  64, V_M2>;
      def Vtype4xi64  : GroupVectorTypeInfo<nxv4i64,  nxv1i64, nxv4i1,  64, V_M4>;
      def Vtype8xi64  : GroupVectorTypeInfo<nxv8i64,  nxv1i64, nxv8i1,  64, V_M8>;
    }
  }

  defset list<FloatVectorTypeInfo> EPIAllFloatVectors = {
    defset list<FloatVectorTypeInfo> EPINoGroupFloatVectors = {
      def Vtype1xf32 : FloatVectorTypeInfo<nxv1f32, nxv1i1, 32, V_MF2, f32, FPR32>;
      def Vtype2xf32 : FloatVectorTypeInfo<nxv2f32, nxv2i1, 32, V_M1, f32, FPR32>;

      def Vtype1xf64 : FloatVectorTypeInfo<nxv1f64, nxv1i1, 64, V_M1, f64, FPR64>;
    }

    defset list<GroupFloatVectorTypeInfo> EPIGroupFloatVectors = {
      def Vtype4xf32  : GroupFloatVectorTypeInfo<nxv4f32,  nxv2f32, nxv4i1,  32, V_M2, f32, FPR32>;
      def Vtype8xf32  : GroupFloatVectorTypeInfo<nxv8f32,  nxv2f32, nxv8i1,  32, V_M4, f32, FPR32>;
      def Vtype16xf32 : GroupFloatVectorTypeInfo<nxv16f32, nxv2f32, nxv16i1, 32, V_M8, f32, FPR32>;

      def Vtype2xf64  : GroupFloatVectorTypeInfo<nxv2f64,  nxv1f64, nxv2i1,  64, V_M2, f64, FPR64>;
      def Vtype4xf64  : GroupFloatVectorTypeInfo<nxv4f64,  nxv1f64, nxv4i1,  64, V_M4, f64, FPR64>;
      def Vtype8xf64  : GroupFloatVectorTypeInfo<nxv8f64,  nxv1f64, nxv8i1,  64, V_M8, f64, FPR64>;
    }
  }
}

defvar NoGroupVectors = !listconcat(EPINoGroupIntegerVectors, EPINoGroupFloatVectors);

// This functor is used to obtain the int vector type that has the same SEW and
// multiplier as the input parameter type
class GetIntVectorTypeInfo<VectorTypeInfo vti, int ieew>
{
  // Equivalent integer vector type. Eg.
  //   Vtype8xi8 → Vtype8xi8 (identity)
  //   Vtype4xf64 → Vtype4xi64
  VectorTypeInfo Vti =
    !cast<VectorTypeInfo>(
      !subst("xi8", "xi" # ieew,
      !subst("xi16", "xi" # ieew,
      !subst("xi32", "xi" # ieew,
      !subst("xi64", "xi" # ieew,
      !subst("xf32", "xi" # ieew,
      !subst("xf64", "xi" # ieew,
      !cast<string>(vti))))))));
}

// This functor is used to obtain the float vector type that has the same SEW
// and multiplier as the input parameter type
class GetFloatVectorTypeInfo<VectorTypeInfo vti>
{
  // Equivalent float vector type. Eg.
  //   Vtype32xi16 → Vtype32xf16
  //   Vtype2xf32 → Vtype2xf32 (identity)
  VectorTypeInfo FVti =
    !cast<VectorTypeInfo>(
      !subst("i", "f", !cast<string>(vti))
    );
}

defset list<VectorTypeInfoToWide> AllWideableIntVectors = {
  def : VectorTypeInfoToWide<Vtype1xi8,   Vtype1xi16>;
  def : VectorTypeInfoToWide<Vtype2xi8,   Vtype2xi16>;
  def : VectorTypeInfoToWide<Vtype4xi8,   Vtype4xi16>;
  def : VectorTypeInfoToWide<Vtype8xi8,   Vtype8xi16>;
  def : VectorTypeInfoToWide<Vtype16xi8,  Vtype16xi16>;
  def : VectorTypeInfoToWide<Vtype32xi8,  Vtype32xi16>;

  def : VectorTypeInfoToWide<Vtype1xi16,  Vtype1xi32>;
  def : VectorTypeInfoToWide<Vtype2xi16,  Vtype2xi32>;
  def : VectorTypeInfoToWide<Vtype4xi16,  Vtype4xi32>;
  def : VectorTypeInfoToWide<Vtype8xi16,  Vtype8xi32>;
  def : VectorTypeInfoToWide<Vtype16xi16, Vtype16xi32>;

  def : VectorTypeInfoToWide<Vtype1xi32,  Vtype1xi64>;
  def : VectorTypeInfoToWide<Vtype2xi32,  Vtype2xi64>;
  def : VectorTypeInfoToWide<Vtype4xi32,  Vtype4xi64>;
  def : VectorTypeInfoToWide<Vtype8xi32,  Vtype8xi64>;
// FIXME what about these?
//def : VectorTypeInfoToWide<Vtype1xi64,  /* FIXME Vtype1xi128 */ i1>;
//def : VectorTypeInfoToWide<Vtype2xi64,  /* FIXME Vtype2xi128 */ i1>;
//def : VectorTypeInfoToWide<Vtype4xi64,  /* FIXME Vtype4xi128 */ i1>;
}

defset list<VectorTypeInfoToWide> AllWideableIntToFloatVectors = {
  def : VectorTypeInfoToWide<Vtype1xi16,  Vtype1xi32>;
  def : VectorTypeInfoToWide<Vtype2xi16,  Vtype2xi32>;
  def : VectorTypeInfoToWide<Vtype4xi16,  Vtype4xi32>;
  def : VectorTypeInfoToWide<Vtype8xi16,  Vtype8xi32>;
  def : VectorTypeInfoToWide<Vtype16xi16, Vtype16xi32>;

  def : VectorTypeInfoToWide<Vtype1xi32,  Vtype1xi64>;
  def : VectorTypeInfoToWide<Vtype2xi32,  Vtype2xi64>;
  def : VectorTypeInfoToWide<Vtype4xi32,  Vtype4xi64>;
  def : VectorTypeInfoToWide<Vtype8xi32,  Vtype8xi64>;
}

defset list<FloatVectorTypeInfoToWide> AllWideableFloatVectors = {
  def : FloatVectorTypeInfoToWide<Vtype1xf32, Vtype1xf64>;
  def : FloatVectorTypeInfoToWide<Vtype2xf32, Vtype2xf64>;
  def : FloatVectorTypeInfoToWide<Vtype4xf32, Vtype4xf64>;
  def : FloatVectorTypeInfoToWide<Vtype8xf32, Vtype8xf64>;
// FIXME what about these?
//def : FloatVectorTypeInfoToWide<Vtype1xf64, /* FIXME: Vtype1xf128 */ i1>;
//def : FloatVectorTypeInfoToWide<Vtype2xf64, /* FIXME: Vtype2xf128 */ i1>;
//def : FloatVectorTypeInfoToWide<Vtype4xf64, /* FIXME: Vtype4xf128 */ i1>;
}

class MaskTypeInfo<ValueType Mas, int Sew, LMULInfo MI> {
  ValueType Mask = Mas;
  // {SEW, VLMul} values set a valid VType to deal with this mask type.
  int SEW = Sew; // FIXME: computed as ELEN/vscale, to be used for loading/storing partial masks
  // FIXME: can we remove this?
  LMULInfo MInfo = MI; // Convention: Minimum VLMul where this mask type appears.
}

defset list<MaskTypeInfo> EPIAllMasks = {
  def : MaskTypeInfo<nxv1i1,  64, V_M1>; // FIXME SEW should be ELEN/1
  def : MaskTypeInfo<nxv2i1,  32, V_M1>; // FIXME SEW should be ELEN/2
  def : MaskTypeInfo<nxv4i1,  16, V_M1>; // FIXME SEW should be ELEN/4
  def : MaskTypeInfo<nxv8i1,  8,  V_M1>;  // FIXME SEW should be ELEN/8
  def : MaskTypeInfo<nxv16i1, 8,  V_M2>;  // FIXME SEW should be ELEN/16, SEW < 8
  def : MaskTypeInfo<nxv32i1, 8,  V_M4>;  // FIXME SEW should be ELEN/32, SEW < 8
  def : MaskTypeInfo<nxv64i1, 8,  V_M8>;  // FIXME SEW should be ELEN/64, SEW < 8
}

defvar AllEEW = [8, 16, 32, 64];

def EPIIntrClassID : GenericEnum {
  let FilterClass = "EPIIntrinsicClassID";
}

def EPIIntrinsicsTable : GenericTable {
  let FilterClass = "EPIIntrinsic";
  let CppTypeName = "EPIIntrinsicInfo";
  let Fields = [ "IntrinsicID", "ClassID", "ExtendOperand", "MaskOperand",
                 "GVLOperand" ];
  let PrimaryKey = [ "IntrinsicID" ];
  let PrimaryKeyName = "getEPIIntrinsicInfo";

  string TypeOf_ClassID = "EPIIntrClassID";
}

class EPIPseudo {
  Pseudo Pseudo = !cast<Pseudo>(NAME);
  Instruction BaseInstr;
  bits<8> VLIndex;
  bits<8> SEWIndex;
  bits<8> MergeOpIndex;
  bits<8> MaskOpIndex;
  bits<3> VLMul;
  bits<8> DynamicFlags = -1;
}

def EPIPseudosTable : GenericTable {
  let FilterClass = "EPIPseudo";
  let CppTypeName = "EPIPseudoInfo";
  let Fields = [ "Pseudo", "BaseInstr", "VLIndex", "SEWIndex",
                 "MergeOpIndex", "MaskOpIndex", "VLMul", "DynamicFlags" ];
  let PrimaryKey = [ "Pseudo" ];
  let PrimaryKeyName = "getEPIPseudoInfo";
}


multiclass pseudo_nullary<VReg result_reg_class,
                          LMULInfo MInfo,
                          string constraints> {
  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 3, SEWIndex = 4, MergeOpIndex = 1,
      MaskOpIndex = 2,
      BaseInstr = !cast<Instruction>(!subst("PseudoEPI", "", NAME)) in
    def "_" # MInfo.MX : Pseudo<(outs result_reg_class:$rd),
                              (ins result_reg_class:$merge,
                                   VMaskOp:$vm,
                                   AVL:$vl, ixlenimm:$sew),
                              []>,
                       EPIPseudo;
}

multiclass pseudo_nullary_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _V : pseudo_nullary<m.vrclass, m, "">;
    }
  }
}

multiclass pseudo_unary<VReg result_reg_class,
                        VReg op_reg_class,
                        LMULInfo MInfo,
                        string constraints> {
  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = 1,
      MaskOpIndex = 3,
      BaseInstr = !cast<Instruction>(!subst("PseudoEPI", "", NAME)) in
    def "_" # MInfo.MX : Pseudo<(outs result_reg_class:$rd),
                                (ins result_reg_class:$merge,
                                     op_reg_class:$rs2, VMaskOp:$vm,
                                     AVL:$vl, ixlenimm:$sew),
                                []>,
                         EPIPseudo;
}

// Special case for masking that does not have a merge operand.
multiclass pseudo_unary_nomerge<RegisterClass result_reg_class,
                                RegisterClass op_reg_class,
                                LMULInfo MInfo,
                                string constraints = ""> {
  let Constraints = constraints,
      Uses = [VL, VTYPE], VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
      MaskOpIndex = 2,
      BaseInstr = !cast<Instruction>(!subst("PseudoEPI", "", NAME)) in
    def "_" # MInfo.MX : Pseudo<(outs result_reg_class:$rd),
                                (ins op_reg_class:$rs2, VMaskOp:$vm,
                                     AVL:$vl, ixlenimm:$sew),
                                []>,
                         EPIPseudo;
}

// Special case for masking that does not have a merge operand nor a mask.
multiclass pseudo_unary_nomask<VReg result_reg_class,
                               DAGOperand op_kind,
                               LMULInfo MInfo,
                               string constraints = ""> {
  let Constraints = constraints,
      Uses = [VL, VTYPE], VLIndex = 2, SEWIndex = 3, MergeOpIndex = -1,
      MaskOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("PseudoEPI", "", NAME)) in
    def "_" # MInfo.MX : Pseudo<(outs result_reg_class:$rd),
                            (ins op_kind:$rs2,
                                 AVL:$vl, ixlenimm:$sew),
                            []>,
                     EPIPseudo;
}

multiclass pseudo_unary_v_m {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _M : pseudo_unary<m.vrclass, VR, m, "">;
    }
  }
}

multiclass pseudo_unary_v_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _V : pseudo_unary<m.vrclass, m.vrclass, m, "">;
    }
  }
}

multiclass pseudo_unary_v_v_x_i_nomask
{
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _V : pseudo_unary_nomask<m.vrclass, m.vrclass, m>;
      defm _X : pseudo_unary_nomask<m.vrclass, GPR, m>;
      defm _I : pseudo_unary_nomask<m.vrclass, simm5, m>;
    }
  }
}

multiclass pseudo_unary_v_f_nomask {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _F : pseudo_unary_nomask<m.vrclass, FPR64, m>;
    }
  }
}

multiclass pseudo_unary_w_v {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = m.value in
    {
      defm _V : pseudo_unary<m.wvrclass, m.vrclass, m, constraints>;
    }
  }
}

multiclass pseudo_unary_v_w {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = m.value in
    {
      defm _W : pseudo_unary<m.vrclass, m.wvrclass, m, constraints>;
    }
  }
}

multiclass pseudo_binary<VReg result_reg_class,
                         VReg op1_reg_class,
                         DAGOperand op2_kind,
                         LMULInfo MInfo,
                         string constraints,
                         list<SchedReadWrite> sched = []> {
  let Constraints = Join<[constraints, "$rd = $merge"], ",">.ret,
      Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
      MaskOpIndex = 4,
      BaseInstr = !cast<Instruction>(!subst("PseudoEPI", "", NAME)) in
    def "_" # MInfo.MX : Pseudo<(outs result_reg_class:$rd),
                                (ins result_reg_class:$merge,
                                     op1_reg_class:$rs2, op2_kind:$rs1,
                                     VMaskOp:$vm, AVL:$vl, ixlenimm:$sew),
                                []>,
                         EPIPseudo,
                         Sched<sched>;
}

// Special case for masking that does not have a merge operand nor a mask.
multiclass pseudo_binary_nomask<VReg result_reg_class,
                                VReg op1_reg_class,
                                DAGOperand op2_kind,
                                LMULInfo MInfo,
                                string constraints = ""> {
  let Constraints = constraints,
      Uses = [VL, VTYPE], VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
      MaskOpIndex = -1,
      BaseInstr = !cast<Instruction>(!subst("PseudoEPI", "", NAME)) in
    def "_" # MInfo.MX : Pseudo<(outs result_reg_class:$rd),
                                (ins op1_reg_class:$rs2, op2_kind:$rs1,
                                     AVL:$vl, ixlenimm:$sew),
                                []>,
                         EPIPseudo;
}

multiclass pseudo_binary_mask_in<VReg result_reg_class,
                                 VReg op1_reg_class,
                                 DAGOperand op2_kind,
                                 LMULInfo MInfo,
                                 string constraints> {
  let Constraints = constraints,
      Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
      MaskOpIndex = 3,
      BaseInstr = !cast<Instruction>(!subst("PseudoEPI", "", NAME)) in
    def "_" # MInfo.MX : Pseudo<(outs result_reg_class:$rd),
                                (ins op1_reg_class:$rs2, op2_kind:$rs1,
                                     VMV0:$maskop, AVL:$vl, ixlenimm:$sew),
                                []>,
                         EPIPseudo;
}

multiclass pseudo_binary_v_vv {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0, usesCustomInserter = 1 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VV : pseudo_binary<m.vrclass, m.vrclass, m.vrclass, m, "">;
    }
  }
}

multiclass pseudo_binary_v_vx<bit force_earlyclobber = 0> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = !if(force_earlyclobber, "@earlyclobber $rd", "");
    let VLMul = m.value in
    {
      defm _VX : pseudo_binary<m.vrclass, m.vrclass, GPR, m, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vs {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VS : pseudo_binary<m.vrclass, m.vrclass, m.vrclass, m, "">;
    }
  }
}

multiclass pseudo_binary_w_vs {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VS : pseudo_binary<m.wvrclass, m.vrclass, m.wvrclass, m, "">;
    }
  }
}

multiclass pseudo_binary_v_vm_nomask {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VM : pseudo_binary_nomask<m.vrclass, m.vrclass, VR, m, "">;
    }
  }
}

multiclass pseudo_binary_v_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VV : pseudo_binary<m.vrclass, m.vrclass, m.vrclass, m, "">;
      defm _VX : pseudo_binary<m.vrclass, m.vrclass, GPR, m, "">;
    }
  }
}

multiclass pseudo_binary_v_vx_vi<DAGOperand imm_kind = simm5,
                                 bit force_earlyclobber = 0> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = !if(force_earlyclobber, "@earlyclobber $rd", "");
    let VLMul = m.value in
    {
      defm _VX : pseudo_binary<m.vrclass, m.vrclass, GPR, m, constraints>;
      defm _VI : pseudo_binary<m.vrclass, m.vrclass, imm_kind, m, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vv_vx_vi<DAGOperand imm_kind = simm5,
                                    bit force_earlyclobber = 0,
                                    list<SchedReadWrite> sched_vv = [],
                                    list<SchedReadWrite> sched_vx = [],
                                    list<SchedReadWrite> sched_vi = []> {
                                    // list<SchedReadWrite> sched_vv = [WriteVPUIALU, ReadVPUIALU, ReadVPUIALU, ReadVL, ReadVTYPE],
                                    // list<SchedReadWrite> sched_vx = [WriteVPUIALU, ReadVPUIALU, ReadVPUScalarIALU, ReadVL, ReadVTYPE],
                                    // list<SchedReadWrite> sched_vi = [WriteVPUIALU, ReadVPUIALU, ReadVL, ReadVTYPE]> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = !if(force_earlyclobber, "@earlyclobber $rd", "");
    let VLMul = m.value in
    {
      defm _VV : pseudo_binary<m.vrclass, m.vrclass, m.vrclass, m, constraints, sched_vv>;
      defm _VX : pseudo_binary<m.vrclass, m.vrclass, GPR, m, constraints, sched_vx>;
      defm _VI : pseudo_binary<m.vrclass, m.vrclass, imm_kind, m, constraints, sched_vi>;
    }
  }
}

multiclass pseudo_binary_v_wv_wx_wi<DAGOperand imm_kind = simm5> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = m.value in
    {
      defm _WV : pseudo_binary<m.vrclass, m.wvrclass, m.vrclass,
                               m, constraints>;
      defm _WX : pseudo_binary<m.vrclass, m.wvrclass, GPR,
                               m, constraints>;
      defm _WI : pseudo_binary<m.vrclass, m.wvrclass, imm_kind,
                               m, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vvm_vxm {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VVM : pseudo_binary_mask_in<m.vrclass, m.vrclass, m.vrclass, m, "">;
      defm _VXM : pseudo_binary_mask_in<m.vrclass, m.vrclass, GPR, m, "">;
    }
  }
}

multiclass pseudo_binary_v_vvm_vxm_vim {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VVM : pseudo_binary_mask_in<m.vrclass, m.vrclass, m.vrclass, m, "">;
      defm _VXM : pseudo_binary_mask_in<m.vrclass, m.vrclass, GPR, m, "">;
      defm _VIM : pseudo_binary_mask_in<m.vrclass, m.vrclass, simm5, m, "">;
    }
  }
}

multiclass pseudo_binary_m_vvm_vxm {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      // These instructions generate a mask register so their result is a VR.
      defm _VVM : pseudo_binary_mask_in<VR, m.vrclass, m.vrclass, m, "">;
      defm _VXM : pseudo_binary_mask_in<VR, m.vrclass, GPR, m, "">;
    }
  }
}

multiclass pseudo_binary_m_vvm_vxm_vim {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      // These instructions generate a mask register so their result is a VR.
      defm _VVM : pseudo_binary_mask_in<VR, m.vrclass, m.vrclass, m, "">;
      defm _VXM : pseudo_binary_mask_in<VR, m.vrclass, GPR, m, "">;
      defm _VIM : pseudo_binary_mask_in<VR, m.vrclass, simm5, m, "">;
    }
  }
}

multiclass pseudo_binary_w_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = m.value in
    {
      defm _VV : pseudo_binary<m.wvrclass, m.vrclass, m.vrclass,
                               m, constraints>;
      defm _VX : pseudo_binary<m.wvrclass, m.vrclass, GPR,
                               m, constraints>;
    }
  }
}

multiclass pseudo_binary_w_wv_wx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = m.value in
    {
      defm _WV : pseudo_binary<m.wvrclass, m.wvrclass, m.vrclass,
                               m, constraints>;
      defm _WX : pseudo_binary<m.wvrclass, m.wvrclass, GPR,
                               m, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VV : pseudo_binary<VR, m.vrclass, m.vrclass, m, "">;
      defm _VX : pseudo_binary<VR, m.vrclass, GPR, m, "">;
    }
  }
}

multiclass pseudo_binary_m_vv_vx_nomask {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = m.value in
    {
      defm _VV : pseudo_binary_nomask<VR, m.vrclass, m.vrclass, m, constraints>;
      defm _VX : pseudo_binary_nomask<VR, m.vrclass, GPR, m, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vx_vi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VX : pseudo_binary<VR, m.vrclass, GPR, m, "">;
      defm _VI : pseudo_binary<VR, m.vrclass, simm5, m, "">;
    }
  }
}

multiclass pseudo_binary_m_vv_vx_vi {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VV : pseudo_binary<VR, m.vrclass, m.vrclass, m, "">;
      defm _VX : pseudo_binary<VR, m.vrclass, GPR, m, "">;
      defm _VI : pseudo_binary<VR, m.vrclass, simm5, m, "">;
    }
  }
}

multiclass pseudo_binary_m_vv_vx_vi_nomask {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = m.value in
    {
      defm _VV : pseudo_binary_nomask<VR, m.vrclass, m.vrclass, m, constraints>;
      defm _VX : pseudo_binary_nomask<VR, m.vrclass, GPR, m, constraints>;
      defm _VI : pseudo_binary_nomask<VR, m.vrclass, simm5, m, constraints>;
    }
  }
}

multiclass pseudo_binary_v_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VF : pseudo_binary<m.vrclass, m.vrclass, FPR64, m, "">;
    }
  }
}

multiclass pseudo_binary_v_vfm {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VFM : pseudo_binary_mask_in<m.vrclass, m.vrclass, FPR64, m, "">;
    }
  }
}

multiclass pseudo_binary_v_vv_vf<list<SchedReadWrite> sched_vv = [],
                                 list<SchedReadWrite> sched_vf = []> {
// multiclass pseudo_binary_v_vv_vf<list<SchedReadWrite> sched_vv = [WriteVPUFALU, ReadVPUFALU, ReadVPUFALU, ReadVL, ReadVTYPE],
//                                  list<SchedReadWrite> sched_vf = [WriteVPUFALU, ReadVPUFALU, ReadVPUScalarFALU, ReadVL, ReadVTYPE]> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VV : pseudo_binary<m.vrclass, m.vrclass, m.vrclass, m, "", sched_vv>;
      defm _VF : pseudo_binary<m.vrclass, m.vrclass, FPR64, m, "", sched_vf>;
    }
  }
}

multiclass pseudo_binary_w_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = m.value in
    {
      defm _VV : pseudo_binary<m.wvrclass, m.vrclass, m.vrclass,
                               m, constraints>;
      defm _VF : pseudo_binary<m.wvrclass, m.vrclass, FPR64,
                               m, constraints>;
    }
  }
}

multiclass pseudo_binary_w_wv_wf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    defvar constraints = "@earlyclobber $rd";
    let VLMul = m.value in
    {
      defm _WV : pseudo_binary<m.wvrclass, m.wvrclass, m.vrclass,
                               m, constraints>;
      defm _WF : pseudo_binary<m.wvrclass, m.wvrclass, FPR64,
                               m, constraints>;
    }
  }
}

multiclass pseudo_binary_m_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VF : pseudo_binary<VR, m.vrclass, FPR64, m, "">;
    }
  }
}

multiclass pseudo_binary_m_vv_vf {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VV : pseudo_binary<VR, m.vrclass, m.vrclass, m, "">;
      defm _VF : pseudo_binary<VR, m.vrclass, FPR64, m, "">;
    }
  }
}

multiclass pseudo_ternary<VReg result_reg_class,
                          RegisterClass op1_reg_class,
                          VReg op2_reg_class,
                          LMULInfo MInfo,
                          string constraints,
                          list<SchedReadWrite> sched = []> {
  let Constraints = Join<["$rd = $rs3", constraints], ",">.ret,
      Uses = [VL, VTYPE],
      VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
      MaskOpIndex = 4,
      BaseInstr = !cast<Instruction>(!subst("PseudoEPI", "", NAME)) in
    def "_" # MInfo.MX : Pseudo<(outs result_reg_class:$rd),
                                (ins result_reg_class:$rs3, op1_reg_class:$rs1,
                                     op2_reg_class:$rs2, VMaskOp:$vm,
                                     AVL:$vl, ixlenimm:$sew),
                                []>,
                         EPIPseudo,
                         Sched<sched>;
}

multiclass pseudo_ternary_v_vv_vx {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VV : pseudo_ternary<m.vrclass, m.vrclass, m.vrclass, m, "">;
      defm _VX : pseudo_ternary<m.vrclass, GPR, m.vrclass, m, "">;
    }
  }
}

multiclass pseudo_ternary_v_vv_vf<list<SchedReadWrite> sched_vv = [],
                                  list<SchedReadWrite> sched_vf = []> {
// multiclass pseudo_ternary_v_vv_vf<list<SchedReadWrite> sched_vv = [WriteVPUFALU, ReadVPUFALU, ReadVPUFALU, ReadVPUFALU, ReadVL, ReadVTYPE],
//                                   list<SchedReadWrite> sched_vf = [WriteVPUFALU, ReadVPUFALU, ReadVPUFALU, ReadVPUScalarFALU, ReadVL, ReadVTYPE]> {
  let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in
  foreach m = MxList in
  {
    let VLMul = m.value in
    {
      defm _VV : pseudo_ternary<m.vrclass, m.vrclass, m.vrclass, m, "", sched_vv>;
      defm _VF : pseudo_ternary<m.vrclass, FPR64, m.vrclass, m, "", sched_vf>;
    }
  }
}

multiclass pseudo_unary_x_m {
  // A different pseudo is defined for each VLMul value, but a VR operand
  // is used in all of them.
  foreach m = MxList in
  {
    let mayLoad = 0, mayStore = 0, hasSideEffects = 0, VLMul = m.value in
    {
      defm _M : pseudo_unary_nomerge<GPR, VR, m>;
    }
  }
}

multiclass pseudo_binary_m_mm {
  // A different pseudo is defined for each VLMul value, but VR operands
  // are used in all of them.
  foreach m = MxList in
  {
    let mayLoad = 0, mayStore = 0, hasSideEffects = 0, VLMul = m.value in
    {
      defm _MM : pseudo_binary_nomask<VR, VR, VR, m>;
    }
  }
}

// Helper to handle the fractional LMUL.
class LMULHandler<int vlmul_>
{
  bit Fractional = !le(5, vlmul_);
  int VLMul = !if(Fractional, !shl(1, !sub(8, vlmul_)), !shl(1, vlmul_));
  string VLMul_str = !if(Fractional, "1/" # VLMul, !cast<string>(VLMul));
}

class DoubleVR<RegisterClass evr, int vlmul_>
      : LMULHandler<vlmul_> {
  RegisterClass r = !if(!not(Fractional),
    !cond(!eq(!cast<string>(evr), "VR") : VRM2,
          !eq(!cast<string>(evr), "VRM2") : VRM4,
          !eq(!cast<string>(evr), "VRM4") : VRM8),
    // LMUL < 1
    VR);
}

class FourTimesVR<RegisterClass evr, int vlmul_>
      : LMULHandler<vlmul_> {
  RegisterClass r = !if(!not(Fractional),
     !cond(!eq(!cast<string>(evr), "VR") : VRM4,
           !eq(!cast<string>(evr), "VRM2") : VRM8),
     // LMUL < 1
     !if(!eq(VLMul, 2), VRM2, VR)
  );
}

class EightTimesVR<RegisterClass evr, int vlmul_>
      : LMULHandler<vlmul_> {
  RegisterClass r = !if(!not(Fractional),
     !cond(!eq(!cast<string>(evr), "VR") : VRM8),
     // LMUL < 1
     !if(!eq(VLMul, 2), VRM4,
     !if(!eq(VLMul, 4), VRM2,
     VR))
  );
}

class HalfVR<RegisterClass evr, int vlmul_>
      : LMULHandler<vlmul_> {
  RegisterClass r = 
    !cond(!eq(!cast<string>(evr), "VR") : VR,
          !eq(!cast<string>(evr), "VRM2") : VR,
          !eq(!cast<string>(evr), "VRM4") : VRM2,
          !eq(!cast<string>(evr), "VRM8") : VRM4);
}

class FourthVR<RegisterClass evr, int vlmul_>
      : LMULHandler<vlmul_> {
  RegisterClass r = 
     !cond(!eq(!cast<string>(evr), "VR") : VR,
           !eq(!cast<string>(evr), "VRM2") : VR,
           !eq(!cast<string>(evr), "VRM4") : VR,
           !eq(!cast<string>(evr), "VRM8") : VRM2);
}

class EighthVR<RegisterClass evr, int vlmul_>
      : LMULHandler<vlmul_> {
  RegisterClass r =
     !cond(!eq(!cast<string>(evr), "VR") : VR,
           !eq(!cast<string>(evr), "VRM2") : VR,
           !eq(!cast<string>(evr), "VRM4") : VR,
           !eq(!cast<string>(evr), "VRM8") : VR);
}

// This computes a list of valid effective EEW for a given SEW and LMUL.
// This is used for indexed loads and stores.
class AllowedEEW<int vlmul_, int sew> : LMULHandler<vlmul_> {
  int SEW = sew;
  list<int> allowed_sew =  !filter(current_ew, AllEEW,
           !or(
             !and(!not(Fractional), // LMUL>=1
                  !le(sew, !mul(8, current_ew,  VLMul)),
                  !le(!mul(8, current_ew, VLMul), !mul(64, sew))),
             !and(Fractional, // LMUL<1
                  !le(!mul(sew, VLMul), !mul(8, current_ew)),
                  !le(current_ew, !mul(64, sew, VLMul)))
           ));
}

// Given a LMUL, SEW and a valid EEW compute the relevant register class
// for the index operand given the data register class
class RegisterClassForIndex<int vlmul_, int sew, int eew, RegisterClass vr>
      : LMULHandler<vlmul_>
{
  RegisterClass r = !if(!not(Fractional),
                      // LMUL>=1
                      !if(!eq(sew, eew), vr,
                      !if(!eq(eew, !mul(2, sew)), DoubleVR<vr, vlmul_>.r,
                      !if(!eq(eew, !mul(4, sew)), FourTimesVR<vr, vlmul_>.r,
                      !if(!eq(eew, !mul(8, sew)), EightTimesVR<vr, vlmul_>.r,
                      !if(!eq(!mul(2, eew), sew), HalfVR<vr, vlmul_>.r,
                      !if(!eq(!mul(4, eew), sew), FourthVR<vr, vlmul_>.r,
                      !if(!eq(!mul(8, eew), sew), EighthVR<vr, vlmul_>.r,
                      !cast<RegisterClass>("RegisterClassForIndex invalid|vr="#vr#"|sew="#sew#"|eew="#eew#"|VLMul="#VLMul_str)))))))),
                      // LMUL<1
                      !if(!eq(sew, eew), vr,
                      !if(!eq(eew, !mul(2, sew)), DoubleVR<vr, vlmul_>.r,
                      !if(!eq(eew, !mul(4, sew)), FourTimesVR<vr, vlmul_>.r,
                      !if(!eq(eew, !mul(8, sew)), EightTimesVR<vr, vlmul_>.r,
                      !if(!eq(!mul(2, eew), sew), HalfVR<vr, vlmul_>.r,
                      !if(!eq(!mul(4, eew), sew), FourthVR<vr, vlmul_>.r,
                      !if(!eq(!mul(8, eew), sew), EighthVR<vr, vlmul_>.r,
                      !cast<RegisterClass>("RegisterClassForIndex invalid|vr="#vr#"|sew="#sew#"|eew="#eew#"|VLMul="#VLMul_str))))))))
                    );
}

let Predicates = [HasVInstructions] in {

defm PseudoEPIVADD        : pseudo_binary_v_vv_vx_vi;
defm PseudoEPIVSUB        : pseudo_binary_v_vv_vx;
defm PseudoEPIVRSUB       : pseudo_binary_v_vx_vi;
defm PseudoEPIVMINU       : pseudo_binary_v_vv_vx;
defm PseudoEPIVMIN        : pseudo_binary_v_vv_vx;
defm PseudoEPIVMAXU       : pseudo_binary_v_vv_vx;
defm PseudoEPIVMAX        : pseudo_binary_v_vv_vx;
defm PseudoEPIVAND        : pseudo_binary_v_vv_vx_vi;
defm PseudoEPIVOR         : pseudo_binary_v_vv_vx_vi;
defm PseudoEPIVXOR        : pseudo_binary_v_vv_vx_vi;

defm PseudoEPIVRGATHER    : pseudo_binary_v_vv_vx_vi<uimm5, /* force_earlyclobber */ 1>;
defm PseudoEPIVSLIDEUP    : pseudo_binary_v_vx_vi<uimm5, /* force_earlyclobber */ 1>;
defm PseudoEPIVSLIDEDOWN  : pseudo_binary_v_vx_vi<uimm5>;

defm PseudoEPIVMSEQ       : pseudo_binary_m_vv_vx_vi;
defm PseudoEPIVMSNE       : pseudo_binary_m_vv_vx_vi;
defm PseudoEPIVMSLTU      : pseudo_binary_m_vv_vx;
defm PseudoEPIVMSLT       : pseudo_binary_m_vv_vx;
defm PseudoEPIVMSLEU      : pseudo_binary_m_vv_vx_vi;
defm PseudoEPIVMSLE       : pseudo_binary_m_vv_vx_vi;

defm PseudoEPIVMSGTU      : pseudo_binary_m_vx_vi;
defm PseudoEPIVMSGT       : pseudo_binary_m_vx_vi;

defm PseudoEPIVSADDU      : pseudo_binary_v_vv_vx_vi;
defm PseudoEPIVSADD       : pseudo_binary_v_vv_vx_vi;
defm PseudoEPIVSSUBU      : pseudo_binary_v_vv_vx;
defm PseudoEPIVSSUB       : pseudo_binary_v_vv_vx;
defm PseudoEPIVAADD       : pseudo_binary_v_vv_vx;
// FIXME missing PseudoEPIVAADDU
defm PseudoEPIVSLL        : pseudo_binary_v_vv_vx_vi<uimm5>;
defm PseudoEPIVASUB       : pseudo_binary_v_vv_vx;
defm PseudoEPIVSMUL       : pseudo_binary_v_vv_vx;
defm PseudoEPIVSRL        : pseudo_binary_v_vv_vx_vi<uimm5>;
defm PseudoEPIVSRA        : pseudo_binary_v_vv_vx_vi<uimm5>;
defm PseudoEPIVSSRL       : pseudo_binary_v_vv_vx_vi<uimm5>;
defm PseudoEPIVSSRA       : pseudo_binary_v_vv_vx_vi<uimm5>;

defm PseudoEPIVNSRL       : pseudo_binary_v_wv_wx_wi<uimm5>;
defm PseudoEPIVNSRA       : pseudo_binary_v_wv_wx_wi<uimm5>;

//FIXME missing PseudoEPIVNCLIP (uimm5)
//FIXME missing PseudoEPIVNCLIPU (uimm5)

defm PseudoEPIVMV_V       : pseudo_unary_v_v_x_i_nomask;
defm PseudoEPIVMERGE      : pseudo_binary_v_vvm_vxm_vim;

// FIXME EDIV instructions disabled
//defm PseudoEPIVDOTU       : pseudo_binary_v_vv;
//defm PseudoEPIVDOT        : pseudo_binary_v_vv;

defm PseudoEPIVREDSUM     : pseudo_binary_v_vs;
defm PseudoEPIVREDAND     : pseudo_binary_v_vs;
defm PseudoEPIVREDOR      : pseudo_binary_v_vs;
defm PseudoEPIVREDXOR     : pseudo_binary_v_vs;
defm PseudoEPIVREDMINU    : pseudo_binary_v_vs;
defm PseudoEPIVREDMIN     : pseudo_binary_v_vs;
defm PseudoEPIVREDMAXU    : pseudo_binary_v_vs;
defm PseudoEPIVREDMAX     : pseudo_binary_v_vs;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, Uses = [VL, VTYPE], VLMul = 0b000 in
foreach m = MxList in
{
  let VLMul = m.value in
  {
    let VLIndex = -1, SEWIndex = 2, MergeOpIndex = -1, BaseInstr = VMV_X_S, 
        MaskOpIndex = -1 in
      def PseudoEPIVMV_X_S_ # m.MX : Pseudo<(outs GPR:$rd),
                                 (ins m.vrclass:$rs2, ixlenimm:$sew),
                                 []>,
                          EPIPseudo;
    let Constraints = "$rd = $merge", VLIndex = 3, SEWIndex = 4,
        MergeOpIndex = -1, BaseInstr = VMV_S_X, MaskOpIndex = -1 in
      def PseudoEPIVMV_S_X_ # m.MX : Pseudo<(outs m.vrclass:$rd),
                                 (ins m.vrclass:$merge, GPR:$rs1, AVL:$vl,
                                      ixlenimm:$sew),
                                 []>,
                          EPIPseudo;
  }
}

defm PseudoEPIVSLIDE1UP   : pseudo_binary_v_vx</* force_earlyclobber */ 1>;
defm PseudoEPIVSLIDE1DOWN : pseudo_binary_v_vx;

defm PseudoEPIVDIVU       : pseudo_binary_v_vv_vx;
defm PseudoEPIVDIV        : pseudo_binary_v_vv_vx;
defm PseudoEPIVREMU       : pseudo_binary_v_vv_vx;
defm PseudoEPIVREM        : pseudo_binary_v_vv_vx;
defm PseudoEPIVMULHU      : pseudo_binary_v_vv_vx;
defm PseudoEPIVMUL        : pseudo_binary_v_vv_vx;
defm PseudoEPIVMULHSU     : pseudo_binary_v_vv_vx;
defm PseudoEPIVMULH       : pseudo_binary_v_vv_vx;

defm PseudoEPIVMADD       : pseudo_ternary_v_vv_vx;
defm PseudoEPIVNMSUB      : pseudo_ternary_v_vv_vx;
defm PseudoEPIVMACC       : pseudo_ternary_v_vv_vx;
defm PseudoEPIVNMSAC      : pseudo_ternary_v_vv_vx;

defm PseudoEPIVWADDU      : pseudo_binary_w_vv_vx;
defm PseudoEPIVWADDU      : pseudo_binary_w_wv_wx;
defm PseudoEPIVWADD       : pseudo_binary_w_vv_vx;
defm PseudoEPIVWADD       : pseudo_binary_w_wv_wx;
defm PseudoEPIVWSUBU      : pseudo_binary_w_vv_vx;
defm PseudoEPIVWSUBU      : pseudo_binary_w_wv_wx;
defm PseudoEPIVWSUB       : pseudo_binary_w_vv_vx;
defm PseudoEPIVWSUB       : pseudo_binary_w_wv_wx;

defm PseudoEPIVADC        : pseudo_binary_v_vvm_vxm_vim;
defm PseudoEPIVMADC       : pseudo_binary_m_vvm_vxm_vim;
defm PseudoEPIVMADC       : pseudo_binary_m_vv_vx_vi_nomask;

defm PseudoEPIVSBC        : pseudo_binary_v_vvm_vxm;
defm PseudoEPIVMSBC       : pseudo_binary_m_vvm_vxm;
defm PseudoEPIVMSBC       : pseudo_binary_m_vv_vx_nomask;

defm PseudoEPIVCOMPRESS   : pseudo_binary_v_vm_nomask;

defm PseudoEPIVWMULU      : pseudo_binary_w_vv_vx;
defm PseudoEPIVWMULSU     : pseudo_binary_w_vv_vx;
defm PseudoEPIVWMUL       : pseudo_binary_w_vv_vx;

defm PseudoEPIVWREDSUM    : pseudo_binary_w_vs;
defm PseudoEPIVWREDSUMU   : pseudo_binary_w_vs;

defm PseudoEPIVFADD       : pseudo_binary_v_vv_vf;
defm PseudoEPIVFREDUSUM   : pseudo_binary_v_vs;
defm PseudoEPIVFSUB       : pseudo_binary_v_vv_vf;
defm PseudoEPIVFREDOSUM   : pseudo_binary_v_vs;
defm PseudoEPIVFMIN       : pseudo_binary_v_vv_vf;
defm PseudoEPIVFREDMIN    : pseudo_binary_v_vs;
defm PseudoEPIVFMAX       : pseudo_binary_v_vv_vf;
defm PseudoEPIVFREDMAX    : pseudo_binary_v_vs;
defm PseudoEPIVFSGNJ      : pseudo_binary_v_vv_vf;
defm PseudoEPIVFSGNJN     : pseudo_binary_v_vv_vf;
defm PseudoEPIVFSGNJX     : pseudo_binary_v_vv_vf;

let mayLoad = 0, mayStore = 0, hasSideEffects = 0, Uses = [VL, VTYPE], VLMul = 0b000, MergeOpIndex = -1 in
foreach m = MxList in
{
  let VLMul = m.value in
  {
    let VLIndex = -1, SEWIndex = 2, BaseInstr = VFMV_F_S, MaskOpIndex = -1 in
      def PseudoEPIVFMV_F_S_ # m.MX : Pseudo<(outs FPR64:$rd),
                                  (ins m.vrclass:$rs2, ixlenimm:$sew),
                                  []>,
                           EPIPseudo;
    let Constraints = "$rd = $merge", VLIndex = 3, SEWIndex = 4,
        MergeOpIndex = -1, BaseInstr = VFMV_S_F, MaskOpIndex = -1 in
      def PseudoEPIVFMV_S_F_ # m.MX : Pseudo<(outs m.vrclass:$rd),
                                  (ins m.vrclass:$merge, FPR64:$rs2, AVL:$vl,
                                       ixlenimm:$sew),
                                  []>,
                           EPIPseudo;
  }
}

defm PseudoEPIVMFEQ       : pseudo_binary_m_vv_vf;
defm PseudoEPIVMFLE       : pseudo_binary_m_vv_vf;
defm PseudoEPIVMFLT       : pseudo_binary_m_vv_vf;
defm PseudoEPIVMFNE       : pseudo_binary_m_vv_vf;

defm PseudoEPIVMFGT       : pseudo_binary_m_vf;
defm PseudoEPIVMFGE       : pseudo_binary_m_vf;

defm PseudoEPIVFDIV       : pseudo_binary_v_vv_vf;
defm PseudoEPIVFRDIV      : pseudo_binary_v_vf;
defm PseudoEPIVFMUL       : pseudo_binary_v_vv_vf;

defm PseudoEPIVFMV_V      : pseudo_unary_v_f_nomask;
defm PseudoEPIVFMERGE     : pseudo_binary_v_vfm;

defm PseudoEPIVFMADD      : pseudo_ternary_v_vv_vf;
defm PseudoEPIVFMSUB      : pseudo_ternary_v_vv_vf;
defm PseudoEPIVFMACC      : pseudo_ternary_v_vv_vf;
defm PseudoEPIVFMSAC      : pseudo_ternary_v_vv_vf;

defm PseudoEPIVFNMADD     : pseudo_ternary_v_vv_vf;
defm PseudoEPIVFNMSUB     : pseudo_ternary_v_vv_vf;
defm PseudoEPIVFNMACC     : pseudo_ternary_v_vv_vf;
defm PseudoEPIVFNMSAC     : pseudo_ternary_v_vv_vf;

defm PseudoEPIVFWADD      : pseudo_binary_w_vv_vf;
defm PseudoEPIVFWADD      : pseudo_binary_w_wv_wf;
defm PseudoEPIVFWSUB      : pseudo_binary_w_vv_vf;
defm PseudoEPIVFWSUB      : pseudo_binary_w_wv_wf;
defm PseudoEPIVFWMUL      : pseudo_binary_w_vv_vf;

defm PseudoEPIVFWREDUSUM  : pseudo_binary_w_vs;
defm PseudoEPIVFWREDOSUM  : pseudo_binary_w_vs;

// FIXME EDIV instructions disabled
//defm PseudoEPIVFDOT       : pseudo_binary_v_vv;

defm PseudoEPIVFSQRT      : pseudo_unary_v_v;

defm PseudoEPIVIOTA       : pseudo_unary_v_m;

defm PseudoEPIVFCVT_XU_F  : pseudo_unary_v_v;
defm PseudoEPIVFCVT_X_F   : pseudo_unary_v_v;
defm PseudoEPIVFCVT_F_XU  : pseudo_unary_v_v;
defm PseudoEPIVFCVT_F_X   : pseudo_unary_v_v;

defm PseudoEPIVFWCVT_XU_F : pseudo_unary_w_v;
defm PseudoEPIVFWCVT_X_F  : pseudo_unary_w_v;
defm PseudoEPIVFWCVT_F_XU : pseudo_unary_w_v;
defm PseudoEPIVFWCVT_F_X  : pseudo_unary_w_v;
defm PseudoEPIVFWCVT_F_F  : pseudo_unary_w_v;

defm PseudoEPIVFNCVT_XU_F : pseudo_unary_v_w;
defm PseudoEPIVFNCVT_X_F  : pseudo_unary_v_w;
defm PseudoEPIVFNCVT_F_XU : pseudo_unary_v_w;
defm PseudoEPIVFNCVT_F_X  : pseudo_unary_v_w;
defm PseudoEPIVFNCVT_F_F  : pseudo_unary_v_w;

defm PseudoEPIVID         : pseudo_nullary_v;

defm PseudoEPIVCPOP       : pseudo_unary_x_m;
defm PseudoEPIVFIRST      : pseudo_unary_x_m;

defm PseudoEPIVMANDN      : pseudo_binary_m_mm;
defm PseudoEPIVMAND       : pseudo_binary_m_mm;
defm PseudoEPIVMOR        : pseudo_binary_m_mm;
defm PseudoEPIVMXOR       : pseudo_binary_m_mm;
defm PseudoEPIVMORN       : pseudo_binary_m_mm;
defm PseudoEPIVMNAND      : pseudo_binary_m_mm;
defm PseudoEPIVMNOR       : pseudo_binary_m_mm;
defm PseudoEPIVMXNOR      : pseudo_binary_m_mm;

// Load/store pseudo instructions

foreach eew = AllEEW in
foreach m = MxList in
{
  defvar LInfo = m.MX;
  defvar vreg = m.vrclass;
  defvar vlmul = m.value;
  defvar evr = m.vrclass;

  let mayLoad = 1, mayStore = 0, hasSideEffects = 0,
      VLMul = vlmul,
      Constraints = "$rd = $merge" in
  {
    let Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = 1,
        MaskOpIndex = 3,
        BaseInstr = !cast<Instruction>("VLE" # eew # "_V") in
      def "PseudoEPIVLE" # eew # "_V_" # LInfo
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, VMaskOp:$mask, AVL:$vl,
                    ixlenimm:$sew),
                   []>,
            EPIPseudo,
            // Sched<[WriteVPULoad, ReadMemBase, ReadVL, ReadVTYPE]>;
            Sched<[]>;

    let Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
        MaskOpIndex = 4,
        BaseInstr = !cast<Instruction>("VLSE" # eew # "_V") in
      def "PseudoEPIVLSE" # eew # "_V_" # LInfo
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, GPR:$rs2, VMaskOp:$mask,
                    AVL:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo,
            // Sched<[WriteVPULoadStrided, ReadMemBase, ReadVL, ReadVTYPE]>;
            Sched<[]>;

    foreach ieew = AllowedEEW<vlmul, /* sew= */ eew>.allowed_sew in {
      defvar ievr = RegisterClassForIndex<vlmul, /* sew= */ eew, /* eew */ ieew, evr>.r;
      let Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
          MaskOpIndex = 4,
          BaseInstr = !cast<Instruction>("VLUXEI" # ieew #"_V") in
        def "PseudoEPIVLXE" # eew # "I" # ieew # "_V_" # LInfo
            : Pseudo<(outs evr:$rd),
                     (ins evr:$merge, GPR:$rs1, ievr:$rs2, VMaskOp:$mask,
                      AVL:$vl, ixlenimm:$sew),
                     []>,
              EPIPseudo,
              // Sched<[WriteVPULoadIndexed, ReadMemBase, ReadVL, ReadVTYPE]>;
              Sched<[]>;
    }
  }

  // Extended loads
  let mayLoad = 1, mayStore = 0, hasSideEffects = 0,
      VLMul = vlmul,
      Constraints = "$rd = $merge" in
  {
    let Uses = [VL, VTYPE], DynamicFlags = 4, VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
        MaskOpIndex = 3,
        BaseInstr = !cast<Instruction>("VLE" # eew # "_V") in
      def "PseudoEPIVLE" # eew # "_EXT_V_" # LInfo
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, VMaskOp:$mask, GPR:$flags, AVL:$vl,
                    ixlenimm:$sew),
                   []>,
            EPIPseudo,
            // Sched<[WriteVPULoad, ReadMemBase, ReadVL, ReadVTYPE]>;
            Sched<[]>;

    let Uses = [VL, VTYPE], DynamicFlags = 5, VLIndex = 6, SEWIndex = 7, MergeOpIndex = 1,
        MaskOpIndex = 4,
        BaseInstr = !cast<Instruction>("VLSE" # eew # "_V") in
      def "PseudoEPIVLSE" # eew # "_EXT_V_" # LInfo
          : Pseudo<(outs evr:$rd),
                   (ins evr:$merge, GPR:$rs1, GPR:$rs2, VMaskOp:$mask,
                    GPR:$flags, AVL:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo,
            // Sched<[WriteVPULoadStrided, ReadMemBase, ReadVL, ReadVTYPE]>;
            Sched<[]>;

    foreach ieew = AllowedEEW<vlmul, /* sew= */ eew>.allowed_sew in {
      defvar ievr = RegisterClassForIndex<vlmul, /* sew= */ eew, /* eew */ ieew, evr>.r;
      let Uses = [VL, VTYPE], DynamicFlags = 5, VLIndex = 6, SEWIndex = 7, MergeOpIndex = 1,
          MaskOpIndex = 4,
          BaseInstr = !cast<Instruction>("VLUXEI" # ieew #"_V") in
        def "PseudoEPIVLXE" # eew # "I" # ieew # "_EXT_V_" # LInfo
            : Pseudo<(outs evr:$rd),
                     (ins evr:$merge, GPR:$rs1, ievr:$rs2, VMaskOp:$mask,
                      GPR:$flags, AVL:$vl, ixlenimm:$sew),
                     []>,
              EPIPseudo,
              // Sched<[WriteVPULoadIndexed, ReadMemBase, ReadVL, ReadVTYPE]>;
              Sched<[]>;
    }
  }

  let mayLoad = 0, mayStore = 1, hasSideEffects = 0,
      VLMul = vlmul in
  {
    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
        MaskOpIndex = 2,
        BaseInstr = !cast<Instruction>("VSE" # eew #"_V") in
      def "PseudoEPIVSE" # eew # "_V_" # LInfo
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, VMaskOp:$mask, AVL:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo,
            // Sched<[WriteVPUStore, ReadVPUStoreData, ReadMemBase, ReadVL, ReadVTYPE]>;
            Sched<[]>;

    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
        MaskOpIndex = 3,
        BaseInstr = !cast<Instruction>("VSSE" # eew # "_V") in
      def "PseudoEPIVSSE" # eew # "_V_" # LInfo
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, GPR:$rs2, VMaskOp:$mask, AVL:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo,
            // Sched<[WriteVPUStoreStrided, ReadVPUStoreData, ReadMemBase, ReadVL, ReadVTYPE]>;
            Sched<[]>;

    foreach ieew = AllowedEEW<vlmul, /* sew= */ eew>.allowed_sew in {
      defvar ievr = RegisterClassForIndex<vlmul, /* sew= */ eew, /* eew */ ieew, evr>.r;
      // Masked stores do not have a merge operand as merge is done in memory
      let Uses = [VL, VTYPE],
          VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
          MaskOpIndex = 3,
          BaseInstr = !cast<Instruction>("VSUXEI" # ieew # "_V") in
        def "PseudoEPIVSXE" # eew # "I" # ieew #"_V_" # LInfo
            : Pseudo<(outs),
                     (ins evr:$rd, GPR:$rs1, ievr:$rs2, VMaskOp:$mask, AVL:$vl,
                          ixlenimm:$sew),
                     []>,
              EPIPseudo,
              // Sched<[WriteVPUStoreIndexed, ReadVPUStoreData, ReadMemBase, ReadVL, ReadVTYPE]>;
              Sched<[]>;
    }
  }

  // Extended stores
  let mayLoad = 0, mayStore = 1, hasSideEffects = 0,
      VLMul = vlmul in
  {
    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE], DynamicFlags = 3,
        VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
        MaskOpIndex = 2,
        BaseInstr = !cast<Instruction>("VSE" # eew #"_V") in
      def "PseudoEPIVSE" # eew # "_EXT_V_" # LInfo
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, VMaskOp:$mask, GPR:$flags, AVL:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo,
            // Sched<[WriteVPUStore, ReadVPUStoreData, ReadMemBase, ReadVL, ReadVTYPE]>;
            Sched<[]>;

    // Masked stores do not have a merge operand as merge is done in memory
    let Uses = [VL, VTYPE],
        DynamicFlags = 4, VLIndex = 5, SEWIndex = 6, MergeOpIndex = -1,
        MaskOpIndex = 3,
        BaseInstr = !cast<Instruction>("VSSE" # eew # "_V") in
      def "PseudoEPIVSSE" # eew # "_EXT_V_" # LInfo
          : Pseudo<(outs),
                   (ins evr:$rd, GPR:$rs1, GPR:$rs2, VMaskOp:$mask, GPR:$flags, AVL:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo,
            // Sched<[WriteVPUStoreStrided, ReadVPUStoreData, ReadMemBase, ReadVL, ReadVTYPE]>;
            Sched<[]>;

    foreach ieew = AllowedEEW<vlmul, /* sew= */ eew>.allowed_sew in {
      defvar ievr = RegisterClassForIndex<vlmul, /* sew= */ eew, /* eew */ ieew, evr>.r;
      // Masked stores do not have a merge operand as merge is done in memory
      let Uses = [VL, VTYPE], DynamicFlags = 4,
          VLIndex = 5, SEWIndex = 6, MergeOpIndex = -1,
          MaskOpIndex = 3,
          BaseInstr = !cast<Instruction>("VSUXEI" # ieew # "_V") in
        def "PseudoEPIVSXE" # eew # "I" # ieew #"_EXT_V_" # LInfo
            : Pseudo<(outs),
                     (ins evr:$rd, GPR:$rs1, ievr:$rs2, VMaskOp:$mask, GPR:$flags, AVL:$vl,
                          ixlenimm:$sew),
                     []>,
              EPIPseudo,
              // Sched<[WriteVPUStoreIndexed, ReadVPUStoreData, ReadMemBase, ReadVL, ReadVTYPE]>;
              Sched<[]>;
    }
  }
}

//===----------------------------------------------------------------------===//
// Pseudo instructions we need for COPY LMUL>1
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1,
    MergeOpIndex = -1, VLIndex = -1, SEWIndex = -1, MaskOpIndex = -1 in
{
  let BaseInstr = VMV2R_V, VLMul = 1 in
  def PseudoEPIVMV2R_M2 : Pseudo<(outs VRM2:$rdest), (ins VRM2:$rs1), []>,
                          EPIPseudo;

  let BaseInstr = VMV4R_V, VLMul = 2 in
  def PseudoEPIVMV4R_M4 : Pseudo<(outs VRM4:$rdest), (ins VRM4:$rs1), []>,
                          EPIPseudo;

  let BaseInstr = VMV8R_V, VLMul = 3 in
  def PseudoEPIVMV8R_M8 : Pseudo<(outs VRM8:$rdest), (ins VRM8:$rs1), []>,
                          EPIPseudo;
}

}

let Predicates = [HasVInstructions] in {

// FIXME: Generalize for LMUL>1

foreach tsize = 2 ... 8 in {

foreach evr = [VR] in
foreach eew = AllEEW in
{
  defvar TupleRegClass = !cast<RegisterClass>("VRN"#tsize#"M1");

  let mayLoad = 1, mayStore = 0, hasSideEffects = 0, VLMul = 0 in
  {
    let Uses = [VL, VTYPE], VLIndex = 4, SEWIndex = 5, MergeOpIndex = 1,
        MaskOpIndex = 3,
        BaseInstr = !cast<Instruction>("VLSEG"#tsize#"E"#eew#"_V") in
      def PseudoEPIVLSEG#tsize#E#eew#_V_M1
          : Pseudo<(outs TupleRegClass:$rd),
                   (ins TupleRegClass:$merge, GPR:$rs1, VMaskOp:$mask, AVL:$vl,
                    ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
        MaskOpIndex = 4,
        BaseInstr = !cast<Instruction>("VLSSEG"#tsize#"E"#eew#"_V") in
      def PseudoEPIVLSSEG#tsize#E#eew#_V_M1
          : Pseudo<(outs TupleRegClass:$rd),
                   (ins TupleRegClass:$merge, GPR:$rs1, GPR:$rs2, VMaskOp:$mask,
                    AVL:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE], VLIndex = 5, SEWIndex = 6, MergeOpIndex = 1,
        MaskOpIndex = 4,
        BaseInstr = !cast<Instruction>("VLUXSEG"#tsize#"EI"#eew#"_V") in
      def PseudoEPIVLXSEG#tsize#EI#eew#_V_M1
          : Pseudo<(outs TupleRegClass:$rd),
                   (ins TupleRegClass:$merge, GPR:$rs1, evr:$rs2, VMaskOp:$mask,
                    AVL:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;
  }

  let mayLoad = 0, mayStore = 1, hasSideEffects = 0, VLMul = 0 in
  {
    let Uses = [VL, VTYPE],
        VLIndex = 3, SEWIndex = 4, MergeOpIndex = -1,
        MaskOpIndex = 2,
        BaseInstr = !cast<Instruction>("VSSEG"#tsize#"E"#eew#"_V") in
      def PseudoEPIVSSEG#tsize#E#eew#_V_M1
          : Pseudo<(outs),
                   (ins TupleRegClass:$rd, GPR:$rs1, VMaskOp:$mask, AVL:$vl,
                        ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE],
        VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
        MaskOpIndex = 3,
        BaseInstr = !cast<Instruction>("VSSSEG"#tsize#"E"#eew#"_V") in
      def PseudoEPIVSSSEG#tsize#E#eew#_V_M1
          : Pseudo<(outs),
                   (ins TupleRegClass:$rd, GPR:$rs1, GPR:$rs2, VMaskOp:$mask,
                        AVL:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;

    let Uses = [VL, VTYPE],
        VLIndex = 4, SEWIndex = 5, MergeOpIndex = -1,
        MaskOpIndex = 3,
        BaseInstr = !cast<Instruction>("VSUXSEG"#tsize#"EI"#eew#"_V") in
      def PseudoEPIVSXSEG#tsize#EI#eew#_V_M1
          : Pseudo<(outs),
                   (ins TupleRegClass:$rd, GPR:$rs1, evr:$rs2, VMaskOp:$mask,
                        AVL:$vl, ixlenimm:$sew),
                   []>,
            EPIPseudo;
  }
}

} // tsize

} // [HasVInstructions]

//===----------------------------------------------------------------------===//
// Patterns. Essential
//===----------------------------------------------------------------------===//

// Floating point instructions with a scalar operand expect such operand to be
// in a register of class FPR64. When dealing with the f32 variant of such
// instructions we need to insert the FPR32 subregister into the FPR64 base
// register to match the instruction operand
class ToFPR64<DAGOperand operand, dag input_dag> {
  dag ret = !if(!eq(!cast<string>(operand),
                    !cast<string>(FPR32)),
                (INSERT_SUBREG (IMPLICIT_DEF), input_dag, sub_32),
                input_dag);
}

// Floating point instructions with a scalar result will generate such result
// in a register of class FPR64. When dealing with the f32 variant of a pattern
// we need to demote the FPR64 base register generated by the instruction to
// the FPR32 subregister expected by the type in the pattern
class FromFPR64<DAGOperand operand, dag input_dag> {
  dag ret = !if(!eq(!cast<string>(operand),
                    !cast<string>(FPR32)),
                (f32 (EXTRACT_SUBREG
                      input_dag,
                      sub_32)),
                input_dag);
}

class FPZero<DAGOperand operand> {
  dag ret = !if(!eq(!cast<string>(operand),
                    !cast<string>(FPR64)),
                (FMV_D_X X0), (FMV_W_X X0));
}

multiclass epi_pat_load_store<ValueType type,
                              ValueType mask_type,
                              int eew,
                              LMULInfo MInfo,
                              VReg reg_class>
{
  // Load
  defvar load_instr_name = "PseudoEPIVLE" # eew # "_V";
  def : Pat<(type (load GPR:$rs1)),
            (!cast<Instruction>(load_instr_name # "_" # MInfo.MX)
             (type (IMPLICIT_DEF)),
             GPR:$rs1,
             (mask_type zero_reg),
             VLMax, eew)>;
  def : Pat<(type (load GPR:$rs1)),
            (!cast<Instruction>(load_instr_name # "_" # MInfo.MX)
             (type (IMPLICIT_DEF)),
             GPR:$rs1,
             (mask_type zero_reg),
             VLMax, eew)>;

  // Store
  defvar store_instr_name = "PseudoEPIVSE" # eew # "_V";
  def : Pat<(store type:$rs2, GPR:$rs1),
            (!cast<Instruction>(store_instr_name # "_" # MInfo.MX)
             reg_class:$rs2, GPR:$rs1,
             (mask_type zero_reg),
             VLMax, eew)>;
  def : Pat<(store type:$rs2, GPR:$rs1),
            (!cast<Instruction>(store_instr_name # "_" # MInfo.MX)
             reg_class:$rs2, GPR:$rs1,
             (mask_type zero_reg),
             VLMax, eew)>;
}

let Predicates = [HasVInstructions] in {
  foreach vti = EPIAllVectors in
  {
    defm : epi_pat_load_store<vti.Vector, vti.Mask, vti.SEW, vti.MInfo,
                              vti.MInfo.vrclass>;
  }

  foreach vti = EPIAllIntegerVectors in {
    def : Pat<(vti.Vector (zext (vti.Mask V0))),
              (!cast<Instruction>("PseudoEPIVMERGE_VIM_"#vti.MInfo.MX)
               (!cast<Instruction>("PseudoEPIVMV_V_I_"#vti.MInfo.MX) 0,
                VLMax, vti.SEW),
               1,
               (vti.Mask V0),
               VLMax, vti.SEW)>;
    // No way to do 'anyext' for masks, so we do 'zext' instead.
    def : Pat<(vti.Vector (anyext (vti.Mask V0))),
              (!cast<Instruction>("PseudoEPIVMERGE_VIM_" # vti.MInfo.MX)
               (!cast<Instruction>("PseudoEPIVMV_V_I_" # vti.MInfo.MX) 0,
                VLMax, vti.SEW),
               1,
               (vti.Mask V0),
               VLMax, vti.SEW)>;
    def : Pat<(vti.Mask (trunc (vti.Vector vti.MInfo.vrclass:$rs1))),
              (!cast<Instruction>("PseudoEPIVMSNE_VV_" # vti.MInfo.MX)
               (vti.Mask (IMPLICIT_DEF)),
               (!cast<Instruction>("PseudoEPIVAND_VI_" # vti.MInfo.MX)
                (vti.Vector (IMPLICIT_DEF)),
                (vti.Vector vti.MInfo.vrclass:$rs1), 1,
                (vti.Mask zero_reg),
                VLMax, vti.SEW),
               (!cast<Instruction>("PseudoEPIVMV_V_I_" # vti.MInfo.MX) 0,
                VLMax, vti.SEW),
               (vti.Mask zero_reg),
               VLMax, vti.SEW)>;
  }
}

//===----------------------------------------------------------------------===//
// Patterns. Common
//===----------------------------------------------------------------------===//

def pow2uimm5 : ImmLeaf<XLenVT, [{
    return isPowerOf2_64(Imm) && isUInt<5>(Log2_64(Imm));
}]>;
def Log2 : SDNodeXForm<imm,
[{
    return CurDAG->getTargetConstant(Log2_64(N->getZExtValue()), SDLoc(N),
        MVT::i64);
}]>;

let Predicates = [HasVInstructions] in {

def : Pat<(int_vscale), (PseudoEPIVSCALE)>;

// FIXME: These patterns can be improved.
def : Pat<(vscale 1), (PseudoEPIVSCALE)>;
def : Pat<(vscale -1), (SUB X0, (PseudoEPIVSCALE))>;
def : Pat<(vscale pow2uimm5:$mul),
          (SLLI (PseudoEPIVSCALE), (Log2 imm:$mul))>;
def : Pat<(vscale (i64 GPR:$imm)),
          (MUL (PseudoEPIVSCALE), $imm)>;

}

// X0 has special meaning for vsetvl/vsetvli.
//  rd | rs1 |   AVL value | Effect on vl
//--------------------------------------------------------------
// !X0 |  X0 |       VLMAX | Set vl to VLMAX
//  X0 |  X0 | Value in vl | Keep current vl, just change vtype.

//===----------------------------------------------------------------------===//
// Patterns. Arithmetic
//===----------------------------------------------------------------------===//

class swap_helper<dag Prefix,
                  dag A,
                  dag B,
                  dag Suffix,
                  bit swap> {
   dag Value = !con(
       Prefix,
       !if(swap, B, A),
       !if(swap, A, B),
       Suffix);
}

multiclass pat_intrinsic_binary<string intrinsic_name,
                                string instruction_name,
                                string kind,
                                ValueType result_type,
                                ValueType op1_type,
                                ValueType op2_type,
                                ValueType mask_type,
                                int sew,
                                LMULInfo MInfo,
                                VReg result_reg_class,
                                VReg op1_reg_class,
                                DAGOperand op2_kind,
                                bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name # "_" # kind # "_" # MInfo.MX);

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_kind:$rs2),
                          (i64 AVL:$vl))),
            swap_helper<
              (instruction (result_type (IMPLICIT_DEF))),
              (instruction
               (op1_type op1_reg_class:$rs1)),
              (instruction ToFPR64<op2_kind, (op2_type op2_kind:$rs2)>.ret),
              (instruction
               (mask_type zero_reg),
               AVL:$vl,
               sew),
              swap>.Value>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                          (result_type result_reg_class:$merge),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_kind:$rs2),
                          (mask_type V0),
                          (i64 AVL:$vl))),
            swap_helper<
              (instruction result_reg_class:$merge),
              (instruction
               op1_reg_class:$rs1),
              (instruction ToFPR64<op2_kind, (op2_type op2_kind:$rs2)>.ret),
              (instruction
               (mask_type V0),
               AVL:$vl,
               sew),
              swap>.Value>;
}

multiclass pat_intrinsic_binary_nomask<string intrinsic_name,
                                       string instruction_name,
                                       string kind,
                                       ValueType result_type,
                                       ValueType op1_type,
                                       ValueType op2_type,
                                       ValueType __mask_type,
                                       int sew,
                                       LMULInfo MInfo,
                                       VReg op1_reg_class,
                                       DAGOperand op2_kind,
                                       bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name # "_" # kind # "_" # MInfo.MX);

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_kind:$rs2),
                          (i64 AVL:$vl))),
            swap_helper<
              (instruction), // empty prefix
              (instruction
               (op1_type op1_reg_class:$rs1)),
              (instruction ToFPR64<op2_kind, (op2_type op2_kind:$rs2)>.ret),
              (instruction
               AVL:$vl,
               sew),
              swap>.Value>;
}

multiclass pat_intrinsic_binary_mask_in<string intrinsic_name,
                                        string instruction_name,
                                        string kind,
                                        ValueType result_type,
                                        ValueType op1_type,
                                        ValueType op2_type,
                                        ValueType mask_type,
                                        int sew,
                                        LMULInfo MInfo,
                                        VReg __result_reg_class,
                                        VReg op1_reg_class,
                                        DAGOperand op2_kind,
                                        bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name # "_" # kind # "_" # MInfo.MX);

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_kind:$rs2),
                          (mask_type V0),
                          (i64 AVL:$vl))),
            swap_helper<
              (instruction), // empty prefix
              (instruction
               op1_reg_class:$rs1),
              (instruction ToFPR64<op2_kind, (op2_type op2_kind:$rs2)>.ret),
              (instruction
               (mask_type V0),
               AVL:$vl,
               sew),
              swap>.Value>;
}

multiclass pat_intrinsic_binary_int_v_vv<string intrinsic_name,
                                         string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, vti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_binary_int_v_vv_vx<string intrinsic_name,
                                            string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_v_vx_vi<string intrinsic_name,
                                            string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_v_vv_vx_vi<string intrinsic_name,
                                               string instruction_name,
                                               DAGOperand imm_kind = simm5>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, imm_kind>;
  }
}

multiclass pat_intrinsic_binary_int_v_vvm_vxm<string intrinsic_name,
                                              string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        vti.Vector, vti.Vector, vti.Vector,
                                        vti.Mask, vti.SEW, vti.MInfo,
                                        vti.MInfo.vrclass, vti.MInfo.vrclass,
                                        vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VXM",
                                        vti.Vector, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.MInfo,
                                        vti.MInfo.vrclass, vti.MInfo.vrclass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_v_vvm_vxm_vim<string intrinsic_name,
                                                  string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        vti.Vector, vti.Vector, vti.Vector,
                                        vti.Mask, vti.SEW, vti.MInfo,
                                        vti.MInfo.vrclass, vti.MInfo.vrclass,
                                        vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VXM",
                                        vti.Vector, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.MInfo,
                                        vti.MInfo.vrclass, vti.MInfo.vrclass, GPR>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VIM",
                                        vti.Vector, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.MInfo,
                                        vti.MInfo.vrclass, vti.MInfo.vrclass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_m_vvm_vxm<string intrinsic_name,
                                              string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        vti.Mask, vti.Vector, vti.Vector,
                                        vti.Mask, vti.SEW, vti.MInfo,
                                        vti.MInfo.vrclass, vti.MInfo.vrclass,
                                        vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VXM",
                                        vti.Mask, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.MInfo,
                                        vti.MInfo.vrclass, vti.MInfo.vrclass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_m_vvm_vxm_vim<string intrinsic_name,
                                                  string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        vti.Mask, vti.Vector, vti.Vector,
                                        vti.Mask, vti.SEW, vti.MInfo,
                                        vti.MInfo.vrclass, vti.MInfo.vrclass,
                                        vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VXM",
                                        vti.Mask, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.MInfo,
                                        vti.MInfo.vrclass, vti.MInfo.vrclass, GPR>;
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VIM",
                                        vti.Mask, vti.Vector, XLenVT,
                                        vti.Mask, vti.SEW, vti.MInfo,
                                        vti.MInfo.vrclass, vti.MInfo.vrclass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_m_vv_vx_nomask<string intrinsic_name,
                                                   string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "VV",
                                       vti.Mask, vti.Vector, vti.Vector,
                                       vti.Mask, vti.SEW, vti.MInfo,
                                       vti.MInfo.vrclass, vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "VX",
                                       vti.Mask, vti.Vector, XLenVT,
                                       vti.Mask, vti.SEW, vti.MInfo,
                                       vti.MInfo.vrclass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_m_vv_vx_vi_nomask<string intrinsic_name,
                                                      string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "VV",
                                       vti.Mask, vti.Vector, vti.Vector,
                                       vti.Mask, vti.SEW, vti.MInfo,
                                       vti.MInfo.vrclass, vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "VX",
                                       vti.Mask, vti.Vector, XLenVT,
                                       vti.Mask, vti.SEW, vti.MInfo,
                                       vti.MInfo.vrclass, GPR>;
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "VI",
                                       vti.Mask, vti.Vector, XLenVT,
                                       vti.Mask, vti.SEW, vti.MInfo,
                                       vti.MInfo.vrclass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_v_vs<string intrinsic_name,
                                         string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VS",
                                vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, vti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_binary_int_w_vs<string intrinsic_name,
                                         string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VS",
                                wti.Vector, vti.Vector, wti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, wti.MInfo.vrclass,
                                vti.MInfo.vrclass, wti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_binary_int_w_vv_vx<string intrinsic_name,
                                            string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                wti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, wti.MInfo.vrclass,
                                vti.MInfo.vrclass, vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                wti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, wti.MInfo.vrclass,
                                vti.MInfo.vrclass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_w_vv_vx_lmul1<string intrinsic_name,
                                                  string instruction_name,
                                                  ValueType result_type,
                                                  ValueType op1_type,
                                                  ValueType tmp_type,
                                                  ValueType mask_type,
                                                  int SourceSEW>
{

def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                        (op1_type VR:$rs1),
                        (op1_type VR:$rs2),
                        (i64 AVL:$vl))),
          (EXTRACT_SUBREG
            (!cast<Instruction>(instruction_name # "_VV_M1")
              (tmp_type (IMPLICIT_DEF)),
              (op1_type VR:$rs1),
              ToFPR64<VR, (op1_type VR:$rs2)>.ret,
              (mask_type zero_reg),
              AVL:$vl,
              SourceSEW), sub_vrm1_0)>;

def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                        (op1_type VR:$rs1),
                        (XLenVT GPR:$rs2),
                        (i64 AVL:$vl))),
          (EXTRACT_SUBREG
            (!cast<Instruction>(instruction_name # "_VX_M1")
              (tmp_type (IMPLICIT_DEF)),
              (op1_type VR:$rs1),
              GPR:$rs2,
              (mask_type zero_reg),
              AVL:$vl,
              SourceSEW), sub_vrm1_0)>;
}

multiclass pat_intrinsic_binary_int_v_wv_wx_wi_lmul1<string intrinsic_name,
                                                     string instruction_name,
                                                     ValueType result_type,
                                                     ValueType op1_type,
                                                     ValueType tmp_type,
                                                     ValueType mask_type,
                                                     int DestSEW>
{

def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                        (op1_type VR:$rs1),
                        (op1_type VR:$rs2),
                        (i64 AVL:$vl))),
            (!cast<Instruction>(instruction_name # "_WV_M1")
              (result_type (IMPLICIT_DEF)),
              (tmp_type (INSERT_SUBREG (IMPLICIT_DEF), VR:$rs1, sub_vrm1_0)),
              ToFPR64<VR, (op1_type VR:$rs2)>.ret,
              (mask_type zero_reg),
              AVL:$vl,
              DestSEW)>;

def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                        (op1_type VR:$rs1),
                        (XLenVT GPR:$rs2),
                        (i64 AVL:$vl))),
            (!cast<Instruction>(instruction_name # "_WX_M1")
              (result_type (IMPLICIT_DEF)),
              (tmp_type (INSERT_SUBREG (IMPLICIT_DEF), VR:$rs1, sub_vrm1_0)),
              GPR:$rs2,
              (mask_type zero_reg),
              AVL:$vl,
              DestSEW)>;

def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                        (op1_type VR:$rs1),
                        (XLenVT uimm5:$rs2),
                        (i64 AVL:$vl))),
            (!cast<Instruction>(instruction_name # "_WI_M1")
              (result_type (IMPLICIT_DEF)),
              (tmp_type (INSERT_SUBREG (IMPLICIT_DEF), VR:$rs1, sub_vrm1_0)),
              uimm5:$rs2,
              (mask_type zero_reg),
              AVL:$vl,
              DestSEW)>;
}

multiclass pat_intrinsic_binary_int_w_wv_wx<string intrinsic_name,
                                            string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WV",
                                wti.Vector, wti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, wti.MInfo.vrclass,
                                wti.MInfo.vrclass, vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WX",
                                wti.Vector, wti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, wti.MInfo.vrclass,
                                wti.MInfo.vrclass, GPR>;
  }
}

// This is the narrowing form.
multiclass pat_intrinsic_binary_int_v_wv_wx_wi<string intrinsic_name,
                                               string instruction_name,
                                               DAGOperand imm_kind = simm5>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WV",
                                vti.Vector, wti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                wti.MInfo.vrclass, vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WX",
                                vti.Vector, wti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                wti.MInfo.vrclass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WI",
                                vti.Vector, wti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                wti.MInfo.vrclass, imm_kind>;
  }
}

multiclass pat_intrinsic_binary_int_m_vv_vx_vi<string intrinsic_name,
                                               string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, VR,
                                vti.MInfo.vrclass, vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.MInfo, VR, vti.MInfo.vrclass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.MInfo, VR, vti.MInfo.vrclass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_m_vv_vx<string intrinsic_name,
                                           string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, VR,
                                vti.MInfo.vrclass, vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.MInfo, VR, vti.MInfo.vrclass, GPR>;
  }
}

multiclass pat_intrinsic_binary_int_m_vx_vi<string intrinsic_name,
                                            string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.MInfo, VR, vti.MInfo.vrclass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Mask, vti.Vector, XLenVT, vti.Mask, vti.SEW,
                                vti.MInfo, VR, vti.MInfo.vrclass, simm5>;
  }
}

multiclass pat_intrinsic_binary_int_m_vv_swapped<string intrinsic_name,
                                                 string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Mask, vti.Vector, vti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, VR,
                                vti.MInfo.vrclass, vti.MInfo.vrclass, /* swapped */ 1>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vv<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.MInfo,
                                fvti.MInfo.vrclass, fvti.MInfo.vrclass, fvti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vf<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Vector, fvti.Vector, fvti.Scalar,
                                fvti.Mask, fvti.SEW, fvti.MInfo,
                                fvti.MInfo.vrclass, fvti.MInfo.vrclass,
                                fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vs<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VS",
                                fvti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.MInfo,
                                fvti.MInfo.vrclass, fvti.MInfo.vrclass, fvti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_binary_fp_w_vs<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar fwti = fvtiToFWti.FWti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VS",
                                fwti.Vector, fvti.Vector, fwti.Vector,
                                fvti.Mask, fvti.SEW, fvti.MInfo,
                                fwti.MInfo.vrclass, fvti.MInfo.vrclass, fwti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vv_vf<string intrinsic_name,
                                           string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.MInfo,
                                fvti.MInfo.vrclass, fvti.MInfo.vrclass, fvti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Vector, fvti.Vector, fvti.Scalar,
                                fvti.Mask, fvti.SEW, fvti.MInfo,
                                fvti.MInfo.vrclass, fvti.MInfo.vrclass,
                                fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vvm<string intrinsic_name,
                                         string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VVM",
                                        fvti.Vector, fvti.Vector, fvti.Vector,
                                        fvti.Mask, fvti.SEW,
                                        fvti.MInfo, fvti.MInfo.vrclass,
                                        fvti.MInfo.vrclass, fvti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_binary_fp_v_vfm<string intrinsic_name,
                                         string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_intrinsic_binary_mask_in<intrinsic_name, instruction_name, "VFM",
                                        fvti.Vector, fvti.Vector, fvti.Scalar,
                                        fvti.Mask, fvti.SEW,
                                        fvti.MInfo, fvti.MInfo.vrclass,
                                        fvti.MInfo.vrclass, fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_w_vv_vf<string intrinsic_name,
                                           string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar fwti = fvtiToFWti.FWti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fwti.Vector, fvti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.MInfo,
                                fwti.MInfo.vrclass, fvti.MInfo.vrclass, fvti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fwti.Vector, fvti.Vector, fvti.Scalar,
                                fvti.Mask, fvti.SEW, fvti.MInfo,
                                fwti.MInfo.vrclass, fvti.MInfo.vrclass,
                                fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_w_wv_wf<string intrinsic_name,
                                           string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar fwti = fvtiToFWti.FWti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WV",
                                fwti.Vector, fwti.Vector, fvti.Vector,
                                fvti.Mask, fvti.SEW, fvti.MInfo,
                                fwti.MInfo.vrclass, fwti.MInfo.vrclass, fvti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "WF",
                                fwti.Vector, fwti.Vector, fvti.Scalar,
                                fvti.Mask, fvti.SEW, fvti.MInfo,
                                fwti.MInfo.vrclass, fwti.MInfo.vrclass,
                                fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_m_vv_vf<string intrinsic_name,
                                           string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Mask, fvti.Vector, fvti.Vector, fvti.Mask,
                                fvti.SEW, fvti.MInfo, VR,
                                fvti.MInfo.vrclass, fvti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Mask, fvti.Vector, fvti.Scalar, fvti.Mask,
                                fvti.SEW, fvti.MInfo, VR,
                                fvti.MInfo.vrclass, fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_m_vf<string intrinsic_name,
                                        string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VF",
                                fvti.Mask, fvti.Vector, fvti.Scalar, fvti.Mask,
                                fvti.SEW, fvti.MInfo, VR,
                                fvti.MInfo.vrclass, fvti.ScalarRegClass>;
  }
}

multiclass pat_intrinsic_binary_fp_m_vv_swapped<string intrinsic_name,
                                                string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                fvti.Mask, fvti.Vector, fvti.Vector, fvti.Mask,
                                fvti.SEW, fvti.MInfo, VR,
                                fvti.MInfo.vrclass, fvti.MInfo.vrclass, /* swapped */ 1>;
  }
}

multiclass pat_intrinsic_binary_m_mm<string intrinsic_name,
                                     string instruction_name>
{
  foreach mti = EPIAllMasks in
  {
    defm : pat_intrinsic_binary_nomask<intrinsic_name, instruction_name, "MM",
                                       mti.Mask, mti.Mask, mti.Mask, mti.Mask,
                                       mti.SEW, mti.MInfo, VR, VR>;
  }
}

multiclass pat_intrinsic_ternary<string intrinsic_name,
                                 string instruction_name,
                                 string kind,
                                 ValueType result_type,
                                 ValueType op1_type,
                                 ValueType op2_type,
                                 ValueType mask_type,
                                 int sew,
                                 LMULInfo MInfo,
                                 VReg result_reg_class,
                                 RegisterClass op1_reg_class,
                                 VReg op2_reg_class>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (result_type result_reg_class:$rs3),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (i64 AVL:$vl))),
            (!cast<Instruction>(instruction_name # "_" # kind # "_" # MInfo.MX)
             result_reg_class:$rs3,
             ToFPR64<op1_reg_class, (op1_type op1_reg_class:$rs1)>.ret,
             op2_reg_class:$rs2,
             (mask_type zero_reg),
             AVL:$vl,
             sew)>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                          (result_type result_reg_class:$rs3),
                          (op1_type op1_reg_class:$rs1),
                          (op2_type op2_reg_class:$rs2),
                          (mask_type V0),
                          (i64 AVL:$vl))),
            (!cast<Instruction>(instruction_name # "_" # kind # "_" # MInfo.MX)
             result_reg_class:$rs3,
             ToFPR64<op1_reg_class, (op1_type op1_reg_class:$rs1)>.ret,
             op2_reg_class:$rs2,
             (mask_type V0),
             AVL:$vl,
             sew)>;
}

multiclass pat_intrinsic_ternary_int_v_vv_vx<string intrinsic_name,
                                             string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VV",
                                 vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                                 vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                 vti.MInfo.vrclass, vti.MInfo.vrclass>;
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VX",
                                 vti.Vector, XLenVT, vti.Vector, vti.Mask,
                                 vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                 GPR, vti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_ternary_fp_v_vv_vf<string intrinsic_name,
                                            string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VV",
                                 fvti.Vector, fvti.Vector, fvti.Vector,
                                 fvti.Mask, fvti.SEW, fvti.MInfo,
                                 fvti.MInfo.vrclass, fvti.MInfo.vrclass, fvti.MInfo.vrclass>;
    defm : pat_intrinsic_ternary<intrinsic_name, instruction_name, "VF",
                                 fvti.Vector, fvti.Scalar, fvti.Vector,
                                 fvti.Mask, fvti.SEW, fvti.MInfo,
                                 fvti.MInfo.vrclass, fvti.ScalarRegClass,
                                 fvti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_binary_any_and_int_v_vx<string intrinsic_name,
                                                 string instruction_name>
{
  foreach vti = EPIAllVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, GPR>;
  }
}

multiclass pat_intrinsic_binary_any_and_int_v_vv_vx_vi<string intrinsic_name,
                                                       string instruction_name,
                                                       DAGOperand imm_kind = simm5>
{
  foreach vti = EPIAllVectors in
  {
    defvar ivti = GetIntVectorTypeInfo<vti, vti.SEW>.Vti;

    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VV",
                                vti.Vector, vti.Vector, ivti.Vector, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, vti.MInfo.vrclass>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, imm_kind>;
  }
}

multiclass pat_intrinsic_binary_any_and_int_v_vx_vi<string intrinsic_name,
                                                    string instruction_name,
                                                    DAGOperand imm_kind = simm5>
{
  foreach vti = EPIAllVectors in
  {
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VX",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, GPR>;
    defm : pat_intrinsic_binary<intrinsic_name, instruction_name, "VI",
                                vti.Vector, vti.Vector, XLenVT, vti.Mask,
                                vti.SEW, vti.MInfo, vti.MInfo.vrclass,
                                vti.MInfo.vrclass, imm_kind>;
  }
}

//class ValueTypePair<ValueType ty1, ValueType ty2>
//{
//   ValueType First = ty1;
//   ValueType Second = ty2;
//}

//class ValueTypeVarList<list<ValueTypePair> VList>
//{
//  list<ValueTypePair> Value = VList;
//}

//def SameSizePairs : ValueTypeVarList<
//  [ ValueTypePair<nxv1i32, nxv1f32>,
//    ValueTypePair<nxv1i64, nxv1f64> ]>;

//multiclass pat_conversions<string instruction,
//                           string intrinsic,
//                           list<ValueTypePair> pairs>
//{

//foreach vtp = pairs in
//{
//def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # intrinsic) (vtp.Second VR:$rs1))),
//          (!cast<Instruction>(instruction # "_V") VR:$rs1)>;
//def : Pat<(vtp.First (!cast<Intrinsic>("int_epi_" # intrinsic # "_mask") (vtp.Second VR:$rs1), V0)),
//          (!cast<Instruction>(instruction # "_V_MASK") VR:$rs1, vmask_only_true.Value)>;

//}

//}

multiclass pat_intrinsic_unary<string intrinsic_name,
                               string instruction_name,
                               string kind,
                               ValueType result_type,
                               ValueType op1_type,
                               ValueType mask_type,
                               int sew,
                               LMULInfo MInfo,
                               VReg result_reg_class,
                               VReg op1_reg_class>
{
  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                          (op1_type op1_reg_class:$rs1),
                          (i64 AVL:$vl))),
            (!cast<Instruction>(instruction_name # "_" # kind # "_" # MInfo.MX)
             (result_type (IMPLICIT_DEF)),
             op1_reg_class:$rs1,
             (mask_type zero_reg),
             AVL:$vl, sew)>;

  def : Pat<(result_type (!cast<Intrinsic>(intrinsic_name # "_mask")
                          (result_type result_reg_class:$merge),
                          (op1_type op1_reg_class:$rs1),
                          (mask_type V0),
                          (i64 AVL:$vl))),
            (!cast<Instruction>(instruction_name # "_" # kind # "_" # MInfo.MX)
             result_reg_class:$merge,
             op1_reg_class:$rs1,
             (mask_type V0),
             AVL:$vl, sew)>;
}

multiclass pat_intrinsic_unary_int_w_v<string intrinsic_name,
                                       string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               wti.Vector, vti.Vector, wti.Mask, vti.SEW,
                               vti.MInfo, wti.MInfo.vrclass, vti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_unary_int_v_w<string intrinsic_name,
                                       string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               vti.Vector, wti.Vector, vti.Mask, vti.SEW,
                               vti.MInfo, vti.MInfo.vrclass, wti.MInfo.vrclass>;
  }
}


multiclass pat_intrinsic_unary_fp_w_v<string intrinsic_name,
                                      string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar fwti = fvtiToFWti.FWti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               fwti.Vector, fvti.Vector, fwti.Mask, fvti.SEW,
                               fvti.MInfo, fwti.MInfo.vrclass,
                               fvti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_unary_int_w_fp_v<string intrinsic_name,
                                          string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar iwti = GetIntVectorTypeInfo<fvtiToFWti.FWti, fvtiToFWti.FWti.SEW>.Vti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               iwti.Vector, fvti.Vector, iwti.Mask, fvti.SEW,
                               fvti.MInfo, iwti.MInfo.vrclass,
                               fvti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_unary_fp_w_int_v<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntToFloatVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar fwti = GetFloatVectorTypeInfo<vtiToWti.Wti>.FVti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "V",
                               fwti.Vector, vti.Vector, fwti.Mask, vti.SEW,
                               vti.MInfo, fwti.MInfo.vrclass, vti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_unary_fp_v_w<string intrinsic_name,
                                      string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar fwti = fvtiToFWti.FWti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "W",
                               fvti.Vector, fwti.Vector, fvti.Mask, fvti.SEW,
                               fvti.MInfo, fvti.MInfo.vrclass,
                               fwti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_unary_int_v_fp_w<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntToFloatVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar fwti = GetFloatVectorTypeInfo<vtiToWti.Wti>.FVti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "W",
                               vti.Vector, fwti.Vector, vti.Mask, vti.SEW,
                               vti.MInfo, vti.MInfo.vrclass, fwti.MInfo.vrclass>;
  }
}

multiclass pat_intrinsic_unary_fp_v_int_w<string intrinsic_name,
                                          string instruction_name>
{
  foreach fvtiToFWti = AllWideableFloatVectors in
  {
    defvar fvti = fvtiToFWti.FVti;
    defvar iwti = GetIntVectorTypeInfo<fvtiToFWti.FWti, fvtiToFWti.FWti.SEW>.Vti;

    defm : pat_intrinsic_unary<intrinsic_name, instruction_name, "W",
                               fvti.Vector, iwti.Vector, fvti.Mask, fvti.SEW,
                               fvti.MInfo, fvti.MInfo.vrclass,
                               iwti.MInfo.vrclass>;
  }
}

//// We are missing cases but are of sizes that won't work
//def WidenedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1i64, nxv1f32>,
//    ValueTypePair<nxv1f32, nxv1i16>,
//    ValueTypePair<nxv1f64, nxv1i32> ]>;

//// Narrowing
//// TODO: This list is the inverse of the one above
//def NarrowedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1f32, nxv1i64>,
//    ValueTypePair<nxv1i16, nxv1f32>,
//    ValueTypePair<nxv1i32, nxv1f64> ]>;

//def FloatWidenedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1f64, nxv1f32> ]>;

//def FloatNarrowedSize : ValueTypeVarList<
//  [ ValueTypePair<nxv1f32, nxv1f64> ]>;

let Predicates = [HasVInstructions] in {

defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vadd", "PseudoEPIVADD">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vsub", "PseudoEPIVSUB">;
defm "" : pat_intrinsic_binary_int_v_vx_vi<"int_epi_vrsub", "PseudoEPIVRSUB">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vminu", "PseudoEPIVMINU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmin", "PseudoEPIVMIN">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmaxu", "PseudoEPIVMAXU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmax", "PseudoEPIVMAX">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vand", "PseudoEPIVAND">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vor", "PseudoEPIVOR">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vxor", "PseudoEPIVXOR">;

defm "" : pat_intrinsic_binary_any_and_int_v_vv_vx_vi<"int_epi_vrgather", "PseudoEPIVRGATHER", uimm5>;
defm "" : pat_intrinsic_binary_any_and_int_v_vx_vi<"int_epi_vslideup", "PseudoEPIVSLIDEUP", uimm5>;
defm "" : pat_intrinsic_binary_any_and_int_v_vx_vi<"int_epi_vslidedown", "PseudoEPIVSLIDEDOWN", uimm5>;

defm "" : pat_intrinsic_binary_int_v_vvm_vxm_vim<"int_epi_vadc", "PseudoEPIVADC">;
defm "" : pat_intrinsic_binary_int_m_vv_vx_vi_nomask<"int_epi_vmadc", "PseudoEPIVMADC">;
defm "" : pat_intrinsic_binary_int_m_vvm_vxm_vim<"int_epi_vmadc_carry_in", "PseudoEPIVMADC">;

defm "" : pat_intrinsic_binary_int_v_vvm_vxm<"int_epi_vsbc", "PseudoEPIVSBC">;
defm "" : pat_intrinsic_binary_int_m_vv_vx_nomask<"int_epi_vmsbc", "PseudoEPIVMSBC">;
defm "" : pat_intrinsic_binary_int_m_vvm_vxm<"int_epi_vmsbc_borrow_in", "PseudoEPIVMSBC">;

defm "" : pat_intrinsic_binary_int_m_vv_vx_vi<"int_epi_vmseq", "PseudoEPIVMSEQ">;
defm "" : pat_intrinsic_binary_int_m_vv_vx_vi<"int_epi_vmsne", "PseudoEPIVMSNE">;
defm "" : pat_intrinsic_binary_int_m_vv_vx<"int_epi_vmsltu", "PseudoEPIVMSLTU">;
defm "" : pat_intrinsic_binary_int_m_vv_vx<"int_epi_vmslt", "PseudoEPIVMSLT">;
defm "" : pat_intrinsic_binary_int_m_vv_vx_vi<"int_epi_vmsleu", "PseudoEPIVMSLEU">;
defm "" : pat_intrinsic_binary_int_m_vv_vx_vi<"int_epi_vmsle", "PseudoEPIVMSLE">;

defm "" : pat_intrinsic_binary_int_m_vx_vi<"int_epi_vmsgtu", "PseudoEPIVMSGTU">;
// Select (int_epi_vmsgtu reg:$rs1, reg:$rs2) as (PseudoEPIVMSLTU reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_int_m_vv_swapped<"int_epi_vmsgtu", "PseudoEPIVMSLTU">;

defm "" : pat_intrinsic_binary_int_m_vx_vi<"int_epi_vmsgt", "PseudoEPIVMSGT">;
// Select (int_epi_vmsgt reg:$rs1, reg:$rs2) as (PseudoEPIVMSLT reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_int_m_vv_swapped<"int_epi_vmsgt", "PseudoEPIVMSLT">;

defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsaddu", "PseudoEPIVSADDU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsadd", "PseudoEPIVSADD">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vssubu", "PseudoEPIVSSUBU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vssub", "PseudoEPIVSSUB">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vaadd", "PseudoEPIVAADD">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsll", "PseudoEPIVSLL", uimm5>;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vasub", "PseudoEPIVASUB">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vsmul", "PseudoEPIVSMUL">;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsrl", "PseudoEPIVSRL", uimm5>;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vsra", "PseudoEPIVSRA", uimm5>;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vssrl", "PseudoEPIVSSRL", uimm5>;
defm "" : pat_intrinsic_binary_int_v_vv_vx_vi<"int_epi_vssra", "PseudoEPIVSSRA", uimm5>;

defm "" : pat_intrinsic_binary_int_v_vvm_vxm_vim<"int_epi_vmerge", "PseudoEPIVMERGE">;

defm "" : pat_intrinsic_binary_int_v_wv_wx_wi<"int_epi_vnsrl", "PseudoEPIVNSRL", uimm5>;
defm "" : pat_intrinsic_binary_int_v_wv_wx_wi<"int_epi_vnsra", "PseudoEPIVNSRA", uimm5>;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnclipu", "VNCLIPU", uimm5>;
//defm "" : pat_intrinsic_binary_int_vv_vx_vi<"int_epi_vnclip", "VNCLIP", uimm5>;

// FIXME EDIV patterns disabled
//defm "" : pat_intrinsic_binary_int_v_vv<"int_epi_vdotu", "PseudoEPIVDOTU">;
//defm "" : pat_intrinsic_binary_int_v_vv<"int_epi_vdot", "PseudoEPIVDOT">;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmaccu", "VWSMACCU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmacc", "VWSMACC">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmsacu", "VWSMSACU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwsmsac", "VWSMSAC">;

defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredsum", "PseudoEPIVREDSUM">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredand", "PseudoEPIVREDAND">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredor", "PseudoEPIVREDOR">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredxor", "PseudoEPIVREDXOR">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredminu", "PseudoEPIVREDMINU">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredmin", "PseudoEPIVREDMIN">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredmaxu", "PseudoEPIVREDMAXU">;
defm "" : pat_intrinsic_binary_int_v_vs<"int_epi_vredmax", "PseudoEPIVREDMAX">;

// The 'int.epi.vmv_x_s' intrinsic will have an illegal result type (i8, i16,
// i32) whenever the operand type is nxv8i8, nxv4i16 or nxv2i32. In such cases,
// the intrinsic is legalized programatically using the RISCVISD::VMV_X_S custom
// node, which is then selected using a pattern as usual.
foreach vti = [Vtype1xi64, Vtype2xi64, Vtype4xi64, Vtype8xi64] in
  def : Pat<(int_epi_vmv_x_s (vti.Vector vti.MInfo.vrclass:$rs2)),
            (!cast<Instruction>("PseudoEPIVMV_X_S_" # vti.MInfo.MX) $rs2, vti.SEW)>;
foreach vti = [Vtype2xi32, Vtype4xi16, Vtype8xi8,
               Vtype4xi32,  Vtype8xi16,  Vtype16xi8,
               Vtype8xi32,  Vtype16xi16, Vtype32xi8,
               Vtype16xi32, Vtype32xi16, Vtype64xi8] in
  def : Pat<(riscv_vmv_x_s (vti.Vector vti.MInfo.vrclass:$rs2)),
            (!cast<Instruction>("PseudoEPIVMV_X_S_" # vti.MInfo.MX) $rs2, vti.SEW)>;

foreach vti = EPIAllIntegerVectors in {
  def : Pat<(vti.Vector (int_epi_vmv_s_x (vti.Vector vti.MInfo.vrclass:$merge), GPR:$rs1, AVL:$vl)),
            (!cast<Instruction>("PseudoEPIVMV_S_X_" # vti.MInfo.MX) $merge, $rs1, AVL:$vl, vti.SEW)>;
}

defm "" : pat_intrinsic_binary_any_and_int_v_vx<"int_epi_vslide1up", "PseudoEPIVSLIDE1UP">;
defm "" : pat_intrinsic_binary_any_and_int_v_vx<"int_epi_vslide1down", "PseudoEPIVSLIDE1DOWN">;

foreach mti = EPIAllMasks in {
  def : Pat<(int_epi_vpopc (mti.Mask VR:$rs2), AVL:$vl),
            (!cast<Instruction>("PseudoEPIVCPOP_M_" # mti.MInfo.MX) $rs2,
              (mti.Mask zero_reg), AVL:$vl, mti.SEW)>;
  def : Pat<(int_epi_vpopc_mask (mti.Mask VR:$rs2), (mti.Mask V0), AVL:$vl),
            (!cast<Instruction>("PseudoEPIVCPOP_M_" # mti.MInfo.MX) $rs2,
              (mti.Mask V0), AVL:$vl, mti.SEW)>;

  def : Pat<(int_epi_vfirst (mti.Mask VR:$rs2), AVL:$vl),
            (!cast<Instruction>("PseudoEPIVFIRST_M_"# mti.MInfo.MX) $rs2,
              (mti.Mask zero_reg), AVL:$vl, mti.SEW)>;
  def : Pat<(int_epi_vfirst_mask (mti.Mask VR:$rs2), (mti.Mask V0), AVL:$vl),
            (!cast<Instruction>("PseudoEPIVFIRST_M_"# mti.MInfo.MX) $rs2,
              (mti.Mask V0), AVL:$vl, mti.SEW)>;
}

//def : Pat<(int_epi_vmsbf VR:$rs1),
//        (VMSBF_M $rs1)>;
//def : Pat<(int_epi_vmsbf_mask VR:$rs1, V0),
//        (VMSBF_M_MASK $rs1, vmask_only_true.Value)>;

//def : Pat<(int_epi_vmsif VR:$rs1),
//        (VMSIF_M $rs1)>;
//def : Pat<(int_epi_vmsif_mask VR:$rs1, V0),
//        (VMSIF_M_MASK $rs1, vmask_only_true.Value)>;

//def : Pat<(int_epi_vmsof VR:$rs1),
//        (VMSOF_M $rs1)>;
//def : Pat<(int_epi_vmsof_mask VR:$rs1, V0),
//        (VMSOF_M_MASK $rs1, vmask_only_true.Value)>;

foreach vtp = EPIAllVectors in {
def : Pat<(vtp.Vector (int_epi_vcompress (vtp.Vector vtp.MInfo.vrclass:$rs2),
                                         (vtp.Mask VR:$rs1),
                                         AVL:$vl)),
          (!cast<Instruction>("PseudoEPIVCOMPRESS_VM_" # vtp.MInfo.MX)
                   $rs2, $rs1, AVL:$vl, vtp.SEW)>;
}

defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmandnot", "PseudoEPIVMANDN">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmand", "PseudoEPIVMAND">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmor", "PseudoEPIVMOR">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmxor", "PseudoEPIVMXOR">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmornot", "PseudoEPIVMORN">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmnand", "PseudoEPIVMNAND">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmnor", "PseudoEPIVMNOR">;
defm "" : pat_intrinsic_binary_m_mm<"int_epi_vmxnor", "PseudoEPIVMXNOR">;

defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vdivu", "PseudoEPIVDIVU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vdiv", "PseudoEPIVDIV">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vremu", "PseudoEPIVREMU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vrem", "PseudoEPIVREM">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmulhu", "PseudoEPIVMULHU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmul", "PseudoEPIVMUL">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmulhsu", "PseudoEPIVMULHSU">;
defm "" : pat_intrinsic_binary_int_v_vv_vx<"int_epi_vmulh", "PseudoEPIVMULH">;

defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vmadd", "PseudoEPIVMADD">;
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vnmsub", "PseudoEPIVNMSUB">;
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vmacc", "PseudoEPIVMACC">;
defm "" : pat_intrinsic_ternary_int_v_vv_vx<"int_epi_vnmsac", "PseudoEPIVNMSAC">;

defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwaddu", "PseudoEPIVWADDU">;
defm "" : pat_intrinsic_binary_int_w_wv_wx<"int_epi_vwaddu_w", "PseudoEPIVWADDU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwadd", "PseudoEPIVWADD">;
defm "" : pat_intrinsic_binary_int_w_wv_wx<"int_epi_vwadd_w", "PseudoEPIVWADD">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwsubu", "PseudoEPIVWSUBU">;
defm "" : pat_intrinsic_binary_int_w_wv_wx<"int_epi_vwsubu_w", "PseudoEPIVWSUBU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwsub", "PseudoEPIVWSUB">;
defm "" : pat_intrinsic_binary_int_w_wv_wx<"int_epi_vwsub_w", "PseudoEPIVWSUB">;

defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwmulu", "PseudoEPIVWMULU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwmulsu", "PseudoEPIVWMULSU">;
defm "" : pat_intrinsic_binary_int_w_vv_vx<"int_epi_vwmul", "PseudoEPIVWMUL">;

defm "" : pat_intrinsic_binary_int_w_vs<"int_epi_vwredsumu", "PseudoEPIVWREDSUMU">;
defm "" : pat_intrinsic_binary_int_w_vs<"int_epi_vwredsum", "PseudoEPIVWREDSUM">;

//// FIXME: These patterns are wrong
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmaccu", "VWMACCU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmacc", "VWMACC">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmsacu", "VWMSACU">;
//defm "" : pat_intrinsic_binary_int_vv_vx<"int_epi_vwmsac", "VWMSAC">;

defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfadd", "PseudoEPIVFADD">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredsum", "PseudoEPIVFREDUSUM">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsub", "PseudoEPIVFSUB">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredosum", "PseudoEPIVFREDOSUM">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfmin", "PseudoEPIVFMIN">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredmin", "PseudoEPIVFREDMIN">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfmax", "PseudoEPIVFMAX">;
defm "" : pat_intrinsic_binary_fp_v_vs<"int_epi_vfredmax", "PseudoEPIVFREDMAX">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsgnj", "PseudoEPIVFSGNJ">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsgnjn", "PseudoEPIVFSGNJN">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfsgnjx", "PseudoEPIVFSGNJX">;

foreach fvti = EPIAllFloatVectors in
{
  def : Pat<(fvti.Scalar (int_epi_vfmv_f_s (fvti.Vector fvti.MInfo.vrclass:$rs2))),
             FromFPR64<fvti.ScalarRegClass,
             (!cast<Instruction>("PseudoEPIVFMV_F_S_" # fvti.MInfo.MX) $rs2, fvti.SEW)>.ret>;

  def : Pat<(fvti.Vector (int_epi_vfmv_s_f (fvti.Vector fvti.MInfo.vrclass:$merge),
                          (fvti.Scalar fvti.ScalarRegClass:$rs2), AVL:$vl)),
            (!cast<Instruction>("PseudoEPIVFMV_S_F_" # fvti.MInfo.MX)
             fvti.MInfo.vrclass:$merge,
             ToFPR64<fvti.ScalarRegClass,
                     (fvti.Scalar fvti.ScalarRegClass:$rs2)>.ret,
             AVL:$vl, fvti.SEW)>;
}

defm "" : pat_intrinsic_binary_fp_m_vv_vf<"int_epi_vmfeq", "PseudoEPIVMFEQ">;
defm "" : pat_intrinsic_binary_fp_m_vv_vf<"int_epi_vmfle", "PseudoEPIVMFLE">;
defm "" : pat_intrinsic_binary_fp_m_vv_vf<"int_epi_vmflt", "PseudoEPIVMFLT">;
defm "" : pat_intrinsic_binary_fp_m_vv_vf<"int_epi_vmfne", "PseudoEPIVMFNE">;

defm "" : pat_intrinsic_binary_fp_m_vf<"int_epi_vmfgt", "PseudoEPIVMFGT">;
// Select (int_epi_vmfgt reg:$rs1, reg:$rs2) as (PseudoEPIVMFLT reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_fp_m_vv_swapped<"int_epi_vmfgt", "PseudoEPIVMFLT">;

defm "" : pat_intrinsic_binary_fp_m_vf<"int_epi_vmfge", "PseudoEPIVMFGE">;
// Select (int_epi_vmfge reg:$rs1, reg:$rs2) as (PseudoEPIVMFLE reg:$rs2, reg:$rs1)
defm "" : pat_intrinsic_binary_fp_m_vv_swapped<"int_epi_vmfge", "PseudoEPIVMFLE">;

defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfdiv", "PseudoEPIVFDIV">;
defm "" : pat_intrinsic_binary_fp_v_vf<"int_epi_vfrdiv", "PseudoEPIVFRDIV">;
defm "" : pat_intrinsic_binary_fp_v_vv_vf<"int_epi_vfmul", "PseudoEPIVFMUL">;

// There is no "vfmerge.vvm" so we select vmerge.vvm
defm "" : pat_intrinsic_binary_fp_v_vvm<"int_epi_vfmerge", "PseudoEPIVMERGE">;
defm "" : pat_intrinsic_binary_fp_v_vfm<"int_epi_vfmerge", "PseudoEPIVFMERGE">;

defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmadd", "PseudoEPIVFMADD">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmadd", "PseudoEPIVFNMADD">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmsub", "PseudoEPIVFMSUB">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmsub", "PseudoEPIVFNMSUB">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmacc", "PseudoEPIVFMACC">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmacc", "PseudoEPIVFNMACC">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfmsac", "PseudoEPIVFMSAC">;
defm "" : pat_intrinsic_ternary_fp_v_vv_vf<"int_epi_vfnmsac", "PseudoEPIVFNMSAC">;

defm "" : pat_intrinsic_binary_fp_w_vv_vf<"int_epi_vfwadd", "PseudoEPIVFWADD">;
defm "" : pat_intrinsic_binary_fp_w_wv_wf<"int_epi_vfwadd_w", "PseudoEPIVFWADD">;
defm "" : pat_intrinsic_binary_fp_w_vs<"int_epi_vfwredsum", "PseudoEPIVFWREDUSUM">;
defm "" : pat_intrinsic_binary_fp_w_vv_vf<"int_epi_vfwsub", "PseudoEPIVFWSUB">;
defm "" : pat_intrinsic_binary_fp_w_wv_wf<"int_epi_vfwsub_w", "PseudoEPIVFWSUB">;
defm "" : pat_intrinsic_binary_fp_w_vs<"int_epi_vfwredosum", "PseudoEPIVFWREDOSUM">;
defm "" : pat_intrinsic_binary_fp_w_vv_vf<"int_epi_vfwmul", "PseudoEPIVFWMUL">;
// FIXME EDIV patterns disabled
//defm "" : pat_intrinsic_binary_fp_v_vv<"int_epi_vfdot", "PseudoEPIVFDOT">;

//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwmacc", "VFWMACC">;
//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwnmacc", "VFWNMACC">;
//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwmsac", "VFWMSAC">;
//defm "" : pat_intrinsic_ternary_fp_vv_vf<"int_epi_vfwnmsac", "VFWNMSAC">;

foreach vti = EPIAllFloatVectors in
  def : Pat<(vti.Vector (int_epi_vfsqrt (vti.Vector vti.MInfo.vrclass:$rs2),
                         AVL:$vl)),
            (!cast<Instruction>("PseudoEPIVFSQRT_V_" # vti.MInfo.MX)
             (vti.Vector (IMPLICIT_DEF)),
             $rs2,
             (vti.Mask zero_reg),
             AVL:$vl, vti.SEW)>;

foreach vti = EPIAllFloatVectors in
  def : Pat<(vti.Vector (int_epi_vfsqrt_mask (vti.Vector vti.MInfo.vrclass:$merge),
                         (vti.Vector vti.MInfo.vrclass:$rs2), (vti.Mask V0),
                         AVL:$vl)),
            (!cast<Instruction>("PseudoEPIVFSQRT_V_" # vti.MInfo.MX)
             $merge,
             $rs2,
             (vti.Mask V0),
             AVL:$vl, vti.SEW)>;

//foreach itp = [ nxv1i32, nxv1i64 ] in
//foreach vtp = EPIAllFloatVectors in
//def : Pat<(itp (int_epi_vfclass (vtp.Vector VR:$rs2))),
//          (VFCLASS_V $rs2)>;

//foreach itp = [ nxv1i32, nxv1i64 ] in
//foreach vtp = EPIAllFloatVectors in
//def : Pat<(itp (int_epi_vfclass_mask (vtp.Vector VR:$rs2), V0)),
//          (VFCLASS_V_MASK $rs2, vmask_only_true.Value)>;

foreach vti = EPIAllIntegerVectors in {
  def : Pat<(vti.Vector (int_epi_viota (vti.Mask VR:$rs2), AVL:$vl)),
            (!cast<Instruction>("PseudoEPIVIOTA_M_" # vti.MInfo.MX)
             (vti.Vector (IMPLICIT_DEF)),
             $rs2,
             (vti.Mask zero_reg),
             AVL:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (int_epi_viota_mask (vti.Vector vti.MInfo.vrclass:$merge),
                         (vti.Mask VR:$rs2), (vti.Mask V0), AVL:$vl)),
            (!cast<Instruction>("PseudoEPIVIOTA_M_" # vti.MInfo.MX)
             $merge,
             $rs2,
             (vti.Mask V0),
             AVL:$vl, vti.SEW)>;
}

multiclass pat_conversions_same_sew<string intrinsic_float_to_int,
                                    string instruction_float_to_int,
                                    string intrinsic_int_to_float,
                                    string instruction_int_to_float>
{
  foreach fvti = EPIAllFloatVectors in
  foreach ivti = [GetIntVectorTypeInfo<fvti, fvti.SEW>.Vti] in
  {
    defvar MInfo = fvti.MInfo;

    def : Pat<(ivti.Vector (!cast<Intrinsic>(intrinsic_float_to_int)
                            (fvti.Vector fvti.MInfo.vrclass:$rs2),
                            AVL:$vl)),
              (!cast<Instruction>(instruction_float_to_int # "_" # MInfo.MX)
               (ivti.Vector (IMPLICIT_DEF)),
               $rs2,
               (ivti.Mask zero_reg),
               AVL:$vl, fvti.SEW)>;

    def : Pat<(ivti.Vector (!cast<Intrinsic>(intrinsic_float_to_int # "_mask")
                            (ivti.Vector fvti.MInfo.vrclass:$merge),
                            (fvti.Vector fvti.MInfo.vrclass:$rs2), (fvti.Mask V0),
                            AVL:$vl)),
              (!cast<Instruction>(instruction_float_to_int # "_" # MInfo.MX)
               $merge,
               $rs2,
               (fvti.Mask V0),
               AVL:$vl, fvti.SEW)>;

    def : Pat<(fvti.Vector (!cast<Intrinsic>(intrinsic_int_to_float)
                            (ivti.Vector fvti.MInfo.vrclass:$rs2),
                            AVL:$vl)),
              (!cast<Instruction>(instruction_int_to_float # "_" # MInfo.MX)
               (fvti.Vector (IMPLICIT_DEF)),
               $rs2,
               (fvti.Mask zero_reg),
               AVL:$vl, fvti.SEW)>;

    def : Pat<(fvti.Vector (!cast<Intrinsic>(intrinsic_int_to_float # "_mask")
                            (fvti.Vector fvti.MInfo.vrclass:$merge),
                            (ivti.Vector fvti.MInfo.vrclass:$rs2), (fvti.Mask V0),
                            AVL:$vl)),
              (!cast<Instruction>(instruction_int_to_float # "_" # MInfo.MX)
               $merge,
               $rs2,
               (fvti.Mask V0),
               AVL:$vl, fvti.SEW)>;
  }
}


defm "" : pat_conversions_same_sew<"int_epi_vfcvt_xu_f", "PseudoEPIVFCVT_XU_F_V",
                                   "int_epi_vfcvt_f_xu", "PseudoEPIVFCVT_F_XU_V">;
defm "" : pat_conversions_same_sew<"int_epi_vfcvt_x_f", "PseudoEPIVFCVT_X_F_V",
                                   "int_epi_vfcvt_f_x", "PseudoEPIVFCVT_F_X_V">;

defm : pat_intrinsic_unary_int_w_fp_v<"int_epi_vfwcvt_xu_f", "PseudoEPIVFWCVT_XU_F">;
defm : pat_intrinsic_unary_int_w_fp_v<"int_epi_vfwcvt_x_f", "PseudoEPIVFWCVT_X_F">;

defm : pat_intrinsic_unary_fp_w_int_v<"int_epi_vfwcvt_f_xu", "PseudoEPIVFWCVT_F_XU">;
defm : pat_intrinsic_unary_fp_w_int_v<"int_epi_vfwcvt_f_x", "PseudoEPIVFWCVT_F_X">;

defm : pat_intrinsic_unary_fp_w_v<"int_epi_vfwcvt_f_f", "PseudoEPIVFWCVT_F_F">;

defm : pat_intrinsic_unary_int_v_fp_w<"int_epi_vfncvt_xu_f", "PseudoEPIVFNCVT_XU_F">;
defm : pat_intrinsic_unary_int_v_fp_w<"int_epi_vfncvt_x_f", "PseudoEPIVFNCVT_X_F">;

defm : pat_intrinsic_unary_fp_v_int_w<"int_epi_vfncvt_f_xu", "PseudoEPIVFNCVT_F_XU">;
defm : pat_intrinsic_unary_fp_v_int_w<"int_epi_vfncvt_f_x", "PseudoEPIVFNCVT_F_X">;

defm : pat_intrinsic_unary_fp_v_w<"int_epi_vfncvt_f_f", "PseudoEPIVFNCVT_F_F">;

multiclass pat_intrinsic_binary_int_w_vX0<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defvar MInfo = vti.MInfo;

    def : Pat<(wti.Vector (!cast<Intrinsic>(intrinsic_name)
                           (vti.Vector vti.MInfo.vrclass:$rs1),
                           (i64 AVL:$vl))),
              (!cast<Instruction>(instruction_name # "_" # MInfo.MX)
               (wti.Vector (IMPLICIT_DEF)),
               vti.MInfo.vrclass:$rs1,
               /* rs2 */ X0,
               (wti.Mask zero_reg),
               AVL:$vl, vti.SEW)>;

    def : Pat<(wti.Vector (!cast<Intrinsic>(intrinsic_name # "_mask")
                           (wti.Vector wti.MInfo.vrclass:$merge),
                           (vti.Vector vti.MInfo.vrclass:$rs1),
                           (wti.Mask V0),
                           (i64 AVL:$vl))),
              (!cast<Instruction>(instruction_name # "_" # MInfo.MX)
               wti.MInfo.vrclass:$merge,
               vti.MInfo.vrclass:$rs1,
               /* rs2 */ X0,
               (wti.Mask V0),
               AVL:$vl, vti.SEW)>;
  }
}

multiclass pat_intrinsic_binary_int_v_wX0<string intrinsic_name,
                                          string instruction_name>
{
  foreach vtiToWti = AllWideableIntVectors in
  {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defvar MInfo = vti.MInfo;

    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic_name)
                           (wti.Vector wti.MInfo.vrclass:$rs1),
                           (i64 AVL:$vl))),
              (!cast<Instruction>(instruction_name # "_" # MInfo.MX)
               (vti.Vector (IMPLICIT_DEF)),
               wti.MInfo.vrclass:$rs1,
               /* rs2 */ X0,
               (vti.Mask zero_reg),
               AVL:$vl, vti.SEW)>;

    def : Pat<(vti.Vector (!cast<Intrinsic>(intrinsic_name#"_mask")
                           (vti.Vector vti.MInfo.vrclass:$merge),
                           (wti.Vector wti.MInfo.vrclass:$rs1),
                           (vti.Mask V0),
                           (i64 AVL:$vl))),
              (!cast<Instruction>(instruction_name # "_" # MInfo.MX)
               vti.MInfo.vrclass:$merge,
               wti.MInfo.vrclass:$rs1,
               /* rs2 */ X0,
               (vti.Mask V0),
               AVL:$vl, vti.SEW)>;
  }
}

defm : pat_intrinsic_binary_int_w_vX0<"int_epi_vwcvtu_x_x", "PseudoEPIVWADDU_VX">;
defm : pat_intrinsic_binary_int_w_vX0<"int_epi_vwcvt_x_x", "PseudoEPIVWADD_VX">;

defm : pat_intrinsic_binary_int_v_wX0<"int_epi_vncvt_x_x", "PseudoEPIVNSRL_WX">;

foreach vti = EPIAllIntegerVectors in {
  def : Pat<(vti.Vector (int_epi_vid AVL:$vl)),
            (!cast<Instruction>("PseudoEPIVID_V_" # vti.MInfo.MX)
             (vti.Vector (IMPLICIT_DEF)),
             (vti.Mask zero_reg),
             AVL:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (int_epi_vid_mask (vti.Vector vti.MInfo.vrclass:$merge),
                         (vti.Mask V0), AVL:$vl)),
            (!cast<Instruction>("PseudoEPIVID_V_" # vti.MInfo.MX)
             $merge,
             (vti.Mask V0),
             AVL:$vl, vti.SEW)>;
}

foreach vti = EPIAllIntegerVectors in {
  def : Pat<(vti.Vector (int_epi_vmv_v_x GPR:$rs2, AVL:$vl)),
            (!cast<Instruction>("PseudoEPIVMV_V_X_" # vti.MInfo.MX)
             $rs2,
             AVL:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (int_epi_vmv_v_x simm5:$imm5, AVL:$vl)),
            (!cast<Instruction>("PseudoEPIVMV_V_I_" # vti.MInfo.MX)
             simm5:$imm5,
             AVL:$vl, vti.SEW)>;
}
foreach fvti = EPIAllFloatVectors in {
  def : Pat<(fvti.Vector (int_epi_vfmv_v_f
                         (fvti.Scalar fvti.ScalarRegClass:$rs2), AVL:$vl)),
            (!cast<Instruction>("PseudoEPIVFMV_V_F_" # fvti.MInfo.MX)
             ToFPR64<fvti.ScalarRegClass,
                     (fvti.Scalar fvti.ScalarRegClass:$rs2)>.ret,
             AVL:$vl, fvti.SEW)>;
}
}

//===----------------------------------------------------------------------===//
// Patterns. Full vector arithmetic operations
//===----------------------------------------------------------------------===//

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_binary_v<SDPatternOperator vop,
                            string instruction_name,
                            ValueType result_type,
                            ValueType op_type,
                            ValueType mask_type,
                            int sew,
                            LMULInfo MInfo,
                            VReg op_reg_class,
                            bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name # "_VV_" # MInfo.MX);
  def : Pat<(result_type (vop
                          (op_type op_reg_class:$rs1),
                          (op_type op_reg_class:$rs2))),
            swap_helper<
              (instruction (result_type (IMPLICIT_DEF))),
              (instruction op_reg_class:$rs1),
              (instruction op_reg_class:$rs2),
              (instruction (mask_type zero_reg),
               VLMax,
               sew),
              swap>.Value>;
}

multiclass pat_vop_binary_v_nomask<SDPatternOperator vop,
                                   string instruction_name,
                                   string kind,
                                   ValueType result_type,
                                   ValueType op_type,
                                   int sew,
                                   LMULInfo MInfo,
                                   VReg op_reg_class,
                                   bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name # "_" # kind #"_" # MInfo.MX);
  def : Pat<(result_type (vop
                          (op_type op_reg_class:$rs1),
                          (op_type op_reg_class:$rs2))),
            swap_helper<
              (instruction), // empty prefix
              (instruction op_reg_class:$rs1),
              (instruction op_reg_class:$rs2),
              (instruction
               VLMax,
               sew),
              swap>.Value>;
}

multiclass pat_vop_binary_xif<SDPatternOperator vop,
                              string instruction_name,
                              string kind,
                              ValueType result_type,
                              ValueType op1_type,
                              ValueType op2_type,
                              ValueType mask_type,
                              int sew,
                              LMULInfo MInfo,
                              VReg op_reg_class,
                              DAGOperand op2_kind,
                              bit swap = 0>
{
  defvar instruction = !cast<Instruction>(instruction_name # "_" # kind #"_" # MInfo.MX);
  def : Pat<(result_type (vop
                          (op1_type op_reg_class:$rs1),
                          (op1_type (splat_vector op2_kind:$rs2)))),
            swap_helper<
              (instruction (result_type (IMPLICIT_DEF))),
              (instruction op_reg_class:$rs1),
              (instruction ToFPR64<op2_kind, (op2_type op2_kind:$rs2)>.ret),
              (instruction (mask_type zero_reg),
               VLMax,
               sew),
              swap>.Value>;
}

multiclass pat_vop_binary_int_v_vv_vx<SDPatternOperator vop,
                                      string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            vti.Vector, vti.Vector, vti.Mask,
                            vti.SEW, vti.MInfo,
                            vti.MInfo.vrclass>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VX",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.MInfo,
                              vti.MInfo.vrclass, GPR>;
  }
}

multiclass pat_vop_binary_int_v_vv_vx_vi<SDPatternOperator vop,
                                         string instruction_name,
                                         DAGOperand imm_kind = simm5>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            vti.Vector, vti.Vector, vti.Mask,
                            vti.SEW, vti.MInfo,
                            vti.MInfo.vrclass>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VX",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.MInfo,
                              vti.MInfo.vrclass, GPR>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VI",
                              vti.Vector, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.MInfo,
                              vti.MInfo.vrclass, imm_kind>;
  }
}

multiclass pat_vop_binary_fp_v_vv_vf<SDPatternOperator vop,
                                     string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            fvti.Vector, fvti.Vector, fvti.Mask,
                            fvti.SEW, fvti.MInfo,
                            fvti.MInfo.vrclass>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VF",
                              fvti.Vector, fvti.Vector, fvti.Scalar, fvti.Mask,
                              fvti.SEW, fvti.MInfo,
                              fvti.MInfo.vrclass, fvti.ScalarRegClass>;
  }
}

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_binary_reduction_int<SDPatternOperator vop,
                                        string instruction_name>
{
  foreach vti = EPIAllIntegerVectors in {
    defvar instruction =
      !cast<Instruction>(instruction_name#"_VS_" # vti.MInfo.MX);
    def : Pat<(vop (vti.Vector vti.MInfo.vrclass:$rs2)),
              (!cast<Instruction>("PseudoEPIVMV_X_S_" # vti.MInfo.MX)
               (vti.Vector (instruction
                (vti.Vector (IMPLICIT_DEF)),
                (vti.Vector vti.MInfo.vrclass:$rs2),
                (!cast<Instruction>("PseudoEPIVMV_S_X_" # vti.MInfo.MX)
                 (vti.Vector (IMPLICIT_DEF)),
                 X0,
                 VLMax, vti.SEW),
                (vti.Mask zero_reg),
                VLMax, vti.SEW)),
               vti.SEW)>;
  }
}

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_binary_reduction_fp<SDPatternOperator vop,
                                       string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in {
    defvar instruction =
      !cast<Instruction>(instruction_name#"_VS_" # fvti.MInfo.MX);
    def : Pat<(vop (fvti.Vector fvti.MInfo.vrclass:$rs2)),
              FromFPR64<fvti.ScalarRegClass,
               (!cast<Instruction>("PseudoEPIVFMV_F_S_" # fvti.MInfo.MX)
                (fvti.Vector (instruction
                 (fvti.Vector (IMPLICIT_DEF)),
                 (fvti.Vector fvti.MInfo.vrclass:$rs2),
                 (!cast<Instruction>("PseudoEPIVFMV_S_F_" # fvti.MInfo.MX)
                  (fvti.Vector (IMPLICIT_DEF)),
                  ToFPR64<fvti.ScalarRegClass, FPZero<fvti.ScalarRegClass>.ret>.ret,
                  VLMax, fvti.SEW),
                 (fvti.Mask zero_reg),
                 VLMax, fvti.SEW)),
                fvti.SEW)>.ret>;
  }
}

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_ternary_v<SDPatternOperator vop,
                             string instruction_name,
                             ValueType vector_type,
                             ValueType mask_type,
                             int sew,
                             LMULInfo MInfo,
                             VReg vector_reg_class>
{
  def : Pat<(vector_type (vop
                          (vector_type vector_reg_class:$rs3),
                          (vector_type vector_reg_class:$rs1),
                          (vector_type vector_reg_class:$rs2))),
            (!cast<Instruction>(instruction_name#"_VV_" # MInfo.MX)
             vector_reg_class:$rs3,
             vector_reg_class:$rs1,
             vector_reg_class:$rs2,
             (mask_type zero_reg),
             VLMax,
             sew)>;
}

multiclass pat_vop_ternary_xf<SDPatternOperator vop,
                              string instruction_name,
                              string kind,
                              ValueType vector_type,
                              ValueType op1_type,
                              ValueType mask_type,
                              int sew,
                              LMULInfo MInfo,
                              VReg vector_reg_class,
                              DAGOperand op1_kind>
{
  def : Pat<(vector_type (vop
                          (vector_type vector_reg_class:$rs3),
                          (vector_type (splat_vector op1_kind:$rs1)),
                          (vector_type vector_reg_class:$rs2))),
            (!cast<Instruction>(instruction_name # "_" # kind # "_" # MInfo.MX)
             vector_reg_class:$rs3,
             ToFPR64<op1_kind, (op1_type op1_kind:$rs1)>.ret,
             vector_reg_class:$rs2,
             (mask_type zero_reg),
             VLMax,
             sew)>;
}

multiclass pat_vop_ternary_fp_v_vv_vf<SDPatternOperator vop,
                                      string instruction_name>
{
  foreach fvti = EPIAllFloatVectors in
  {
    defm : pat_vop_ternary_v<vop, instruction_name,
                             fvti.Vector, fvti.Mask,
                             fvti.SEW, fvti.MInfo,
                             fvti.MInfo.vrclass>;
    defm : pat_vop_ternary_xf<vop, instruction_name, "VF",
                              fvti.Vector, fvti.Scalar, fvti.Mask,
                              fvti.SEW, fvti.MInfo,
                              fvti.MInfo.vrclass, fvti.ScalarRegClass>;
  }
}

// FIXME: Add patterns that can select masked versions
multiclass pat_vop_unary<SDPatternOperator vop,
                         string instruction_name,
                         ValueType vector_type,
                         ValueType mask_type,
                         int sew,
                         LMULInfo MInfo,
                         VReg vector_reg_class>
{
  def : Pat<(vector_type (vop
                          (vector_type vector_reg_class:$rs2))),
            (!cast<Instruction>(instruction_name # "_V_" # MInfo.MX)
             (vector_type (IMPLICIT_DEF)),
             vector_reg_class:$rs2,
             (mask_type zero_reg),
             VLMax,
             sew)>;
}

multiclass pat_vop_unary_fp_v_v<SDPatternOperator vop,
                                string instruction_name>
{
  foreach vti = EPIAllFloatVectors in
  {
    defm : pat_vop_unary<vop, instruction_name,
                         vti.Vector, vti.Mask,
                         vti.SEW, vti.MInfo,
                         vti.MInfo.vrclass>;
  }
}

let Predicates = [HasVInstructions] in {

// defm "" : pat_vop_binary_int_v_vv_vx_vi<add, "PseudoEPIVADD">;
// defm "" : pat_vop_binary_int_v_vv_vx_vi<shl, "PseudoEPIVSLL", uimm5>;
// defm "" : pat_vop_binary_int_v_vv_vx_vi<and, "PseudoEPIVAND">;
// defm "" : pat_vop_binary_int_v_vv_vx<sub, "PseudoEPIVSUB">;
defm "" : pat_vop_binary_int_v_vv_vx<mul, "PseudoEPIVMUL">;

defm "" : pat_vop_binary_fp_v_vv_vf<fadd, "PseudoEPIVFADD">;
defm "" : pat_vop_binary_fp_v_vv_vf<fsub, "PseudoEPIVFSUB">;
defm "" : pat_vop_binary_fp_v_vv_vf<fmul, "PseudoEPIVFMUL">;
// FIXME: Our understanding from the spec is that the following is wrong and
// should be 'fminimum' and 'fmaximum' instead. However this issue is found in
// the base operations in F and D so for now we keep it coherent with those.
defm "" : pat_vop_binary_fp_v_vv_vf<fminnum, "PseudoEPIVFMIN">;
defm "" : pat_vop_binary_fp_v_vv_vf<fmaxnum, "PseudoEPIVFMAX">;

defm "" : pat_vop_binary_reduction_fp<vecreduce_fmax, "PseudoEPIVFREDMAX">;
defm "" : pat_vop_binary_reduction_fp<vecreduce_fmin, "PseudoEPIVFREDMIN">;

foreach fvti = EPIAllFloatVectors in
{
  defvar ivti = GetIntVectorTypeInfo<fvti, fvti.SEW>.Vti;

  def : Pat<(fvti.Vector (splat_vector fvti.ScalarRegClass:$value)),
            (!cast<Instruction>("PseudoEPIVFMV_V_F_" # fvti.MInfo.MX)
             ToFPR64<fvti.ScalarRegClass,
                     (fvti.Scalar fvti.ScalarRegClass:$value)>.ret,
             VLMax,
             fvti.SEW)>;

  def : Pat<(fvti.Vector (sint_to_fp (ivti.Vector ivti.MInfo.vrclass:$value))),
            (!cast<Instruction>("PseudoEPIVFCVT_F_X_V_" # fvti.MInfo.MX)
             (fvti.Vector (IMPLICIT_DEF)),
             $value,
             (ivti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;

  def : Pat<(fvti.Vector (uint_to_fp (ivti.Vector ivti.MInfo.vrclass:$value))),
            (!cast<Instruction>("PseudoEPIVFCVT_F_XU_V_" # fvti.MInfo.MX)
             (fvti.Vector (IMPLICIT_DEF)),
             $value,
             (ivti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;

  def : Pat<(ivti.Vector (fp_to_sint (fvti.Vector fvti.MInfo.vrclass:$value))),
            (!cast<Instruction>("PseudoEPIVFCVT_X_F_V_" # fvti.MInfo.MX)
             (ivti.Vector (IMPLICIT_DEF)),
             $value,
             (fvti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;

  def : Pat<(ivti.Vector (fp_to_uint (fvti.Vector fvti.MInfo.vrclass:$value))),
            (!cast<Instruction>("PseudoEPIVFCVT_XU_F_V_" # fvti.MInfo.MX)
             (ivti.Vector (IMPLICIT_DEF)),
             $value,
             (fvti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;

  def : Pat<(fvti.Vector (fneg (fvti.Vector fvti.MInfo.vrclass:$value))),
            (!cast<Instruction>("PseudoEPIVFSGNJN_VV_" # fvti.MInfo.MX)
             (fvti.Vector (IMPLICIT_DEF)),
             $value,
             $value,
             (fvti.Mask zero_reg),
             VLMax,
             fvti.SEW)>;
}

foreach vti = EPIAllIntegerVectors in
{
  def : Pat<(vti.Vector (splat_vector GPR:$value)),
            (!cast<Instruction>("PseudoEPIVMV_V_X_" # vti.MInfo.MX)
             GPR:$value,
             VLMax, vti.SEW)>;
  def : Pat<(vti.Vector (splat_vector simm5:$value)),
            (!cast<Instruction>("PseudoEPIVMV_V_I_" # vti.MInfo.MX)
             simm5:$value,
             VLMax, vti.SEW)>;
}

foreach vti = EPIAllIntegerVectors in {

def : Pat<(vti.Vector (riscv_shuffle_extend
                         (vti.Vector vti.MInfo.vrclass:$rs1),
                         uimm5:$shamt)),
         (!cast<Instruction>("PseudoEPIVRGATHER_VV_" # vti.MInfo.MX)
             (vti.Vector (IMPLICIT_DEF)),
             (vti.Vector vti.MInfo.vrclass:$rs1),
             (!cast<Instruction>("PseudoEPIVSRL_VI_" # vti.MInfo.MX)
                (vti.Vector (IMPLICIT_DEF)),
                (!cast<Instruction>("PseudoEPIVID_V_" # vti.MInfo.MX)
                   (vti.Vector (IMPLICIT_DEF)),
                   (vti.Mask zero_reg),
                   VLMax, vti.SEW),
                (uimm5:$shamt),
                (vti.Mask zero_reg),
                VLMax, vti.SEW),
             (vti.Mask zero_reg),
             VLMax, vti.SEW)>;
}

foreach fvti = EPIAllFloatVectors in {
  def : Pat<(fabs (fvti.Vector fvti.MInfo.vrclass:$rs2)),
            (!cast<Instruction>("PseudoEPIVFSGNJX_VV_" # fvti.MInfo.MX)
             (fvti.Vector (IMPLICIT_DEF)),
             (fvti.Vector fvti.MInfo.vrclass:$rs2),
             (fvti.Vector fvti.MInfo.vrclass:$rs2),
             (fvti.Mask zero_reg),
             VLMax, fvti.SEW)>;
}

}

//===----------------------------------------------------------------------===//
// Patterns. Full vector relational operations
//===----------------------------------------------------------------------===//

multiclass pat_vop_binary_int_m_vv<SDPatternOperator vop,
                                   string instruction_name,
                                   bit swap = 0>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            vti.Mask, vti.Vector, vti.Mask,
                            vti.SEW, vti.MInfo,
                            vti.MInfo.vrclass, swap>;
  }
}

multiclass pat_vop_binary_int_m_vv_vx<SDPatternOperator vop,
                                      string instruction_name,
                                      bit swap = 0>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            vti.Mask, vti.Vector, vti.Mask,
                            vti.SEW, vti.MInfo,
                            vti.MInfo.vrclass, swap>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VX",
                              vti.Mask, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.MInfo,
                              vti.MInfo.vrclass, GPR, swap>;
  }
}

multiclass pat_vop_binary_int_m_vv_vx_vi<SDPatternOperator vop,
                                         string instruction_name,
                                         bit swap = 0>
{
  foreach vti = EPIAllIntegerVectors in
  {
    defm : pat_vop_binary_v<vop, instruction_name,
                            vti.Mask, vti.Vector, vti.Mask,
                            vti.SEW, vti.MInfo,
                            vti.MInfo.vrclass, swap>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VX",
                              vti.Mask, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.MInfo,
                              vti.MInfo.vrclass, GPR, swap>;
    defm : pat_vop_binary_xif<vop, instruction_name, "VI",
                              vti.Mask, vti.Vector, XLenVT, vti.Mask,
                              vti.SEW, vti.MInfo,
                              vti.MInfo.vrclass, simm5, swap>;
  }
}

let Predicates = [HasVInstructions] in {

defm "" : pat_vop_binary_int_m_vv_vx_vi<seteq, "PseudoEPIVMSEQ">;
defm "" : pat_vop_binary_int_m_vv_vx_vi<setne, "PseudoEPIVMSNE">;
defm "" : pat_vop_binary_int_m_vv_vx_vi<setule, "PseudoEPIVMSLEU">;
defm "" : pat_vop_binary_int_m_vv_vx_vi<setle, "PseudoEPIVMSLE">;
defm "" : pat_vop_binary_int_m_vv_vx<setult, "PseudoEPIVMSLTU">;
defm "" : pat_vop_binary_int_m_vv_vx<setlt, "PseudoEPIVMSLT">;
defm "" : pat_vop_binary_int_m_vv<setugt, "PseudoEPIVMSLTU", /* swap */ 1>;
defm "" : pat_vop_binary_int_m_vv<setgt, "PseudoEPIVMSLT", /* swap */ 1>;
defm "" : pat_vop_binary_int_m_vv<setuge, "PseudoEPIVMSLEU", /* swap */ 1>;
defm "" : pat_vop_binary_int_m_vv<setge, "PseudoEPIVMSLE", /* swap */ 1>;

}

//===----------------------------------------------------------------------===//
// Patterns. Full vector mask logical operations
//===----------------------------------------------------------------------===//

multiclass pat_vop_binary_m_mm<SDPatternOperator vop,
                               string instruction_name>
{
  foreach mti = EPIAllMasks in
  {
    defm : pat_vop_binary_v_nomask<vop, instruction_name, "MM",
                                   mti.Mask, mti.Mask,
                                   mti.SEW, mti.MInfo, VR>;
  }
}

let Predicates = [HasVInstructions] in {

defm "" : pat_vop_binary_m_mm<and, "PseudoEPIVMAND">;
defm "" : pat_vop_binary_m_mm<xor, "PseudoEPIVMXOR">;
defm "" : pat_vop_binary_m_mm<or, "PseudoEPIVMOR">;

}

//===----------------------------------------------------------------------===//
// Patterns. Load/store
//===----------------------------------------------------------------------===//


multiclass pat_load {
  foreach vti = EPIAllVectors in {
    defvar MInfo = vti.MInfo;

    def : Pat<(vti.Vector (int_epi_vload GPR:$rs1, AVL:$vl)),
              (!cast<Instruction>("PseudoEPIVLE" # vti.SEW # "_V_" # MInfo.MX)
               (vti.Vector (IMPLICIT_DEF)),
               $rs1,
               (vti.Mask zero_reg),
               AVL:$vl,
               vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_strided GPR:$rs1, GPR:$rs2, AVL:$vl)),
              (!cast<Instruction>("PseudoEPIVLSE" # vti.SEW #"_V_" # MInfo.MX)
               (vti.Vector (IMPLICIT_DEF)),
               $rs1, $rs2,
               (vti.Mask zero_reg),
               AVL:$vl,
               vti.SEW)>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(vti.Vector (int_epi_vload_indexed GPR:$rs1,
                             (ivti.Vector ivti.MInfo.vrclass:$rs2), AVL:$vl)),
                (!cast<Instruction>("PseudoEPIVLXE" # vti.SEW # "I" # ieew # "_V_" # MInfo.MX)
                 (vti.Vector (IMPLICIT_DEF)),
                 $rs1, $rs2,
                 (vti.Mask zero_reg),
                 AVL:$vl,
                 vti.SEW)>;
    }

    def : Pat<(vti.Vector (int_epi_vload_mask (vti.Vector vti.MInfo.vrclass:$merge),
                           GPR:$rs1, (vti.Mask V0), AVL:$vl)),
               (!cast<Instruction>("PseudoEPIVLE" # vti.SEW # "_V_" # MInfo.MX)
                $merge,
                $rs1,
                (vti.Mask V0),
                AVL:$vl,
                vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_strided_mask
                           (vti.Vector vti.MInfo.vrclass:$merge), GPR:$rs1, GPR:$rs2,
                           (vti.Mask V0), AVL:$vl)),
              (!cast<Instruction>("PseudoEPIVLSE" # vti.SEW # "_V_" # MInfo.MX)
               $merge,
               $rs1, $rs2,
               (vti.Mask V0),
               AVL:$vl,
               vti.SEW)>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(vti.Vector (int_epi_vload_indexed_mask
                             (vti.Vector vti.MInfo.vrclass:$merge), GPR:$rs1,
                             (ivti.Vector ivti.MInfo.vrclass:$rs2), (vti.Mask V0),
                             AVL:$vl)),
                (!cast<Instruction>("PseudoEPIVLXE" # vti.SEW # "I" # ieew # "_V_" # MInfo.MX)
                 $merge,
                 $rs1, $rs2,
                 (vti.Mask V0),
                 AVL:$vl,
                 vti.SEW)>;

    }

    defvar nontemporal = !shl(1, 9);
    def : Pat<(vti.Vector (int_epi_vload_nt GPR:$rs1, AVL:$vl)),
              (!cast<Instruction>("PseudoEPIVLE" # vti.SEW # "_V_" # MInfo.MX)
               (vti.Vector (IMPLICIT_DEF)),
               $rs1,
               (vti.Mask zero_reg),
               AVL:$vl,
               !or(vti.SEW, nontemporal))>;

    def : Pat<(vti.Vector (int_epi_vload_nt_strided GPR:$rs1, GPR:$rs2, AVL:$vl)),
              (!cast<Instruction>("PseudoEPIVLSE" # vti.SEW # "_V_" # MInfo.MX)
               (vti.Vector (IMPLICIT_DEF)),
               $rs1, $rs2,
               (vti.Mask zero_reg),
               AVL:$vl,
               !or(vti.SEW, nontemporal))>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(vti.Vector (int_epi_vload_nt_indexed GPR:$rs1,
                             (ivti.Vector ivti.MInfo.vrclass:$rs2), AVL:$vl)),
                (!cast<Instruction>("PseudoEPIVLXE" # vti.SEW # "I" # ieew # "_V_" # MInfo.MX)
                 (vti.Vector (IMPLICIT_DEF)),
                 $rs1, $rs2,
                 (vti.Mask zero_reg),
                 AVL:$vl,
                 !or(vti.SEW, nontemporal))>;
    }

    def : Pat<(vti.Vector (int_epi_vload_nt_mask (vti.Vector vti.MInfo.vrclass:$merge),
                           GPR:$rs1, (vti.Mask V0), AVL:$vl)),
               (!cast<Instruction>("PseudoEPIVLE" # vti.SEW # "_V_" # MInfo.MX)
                $merge,
                $rs1,
                (vti.Mask V0),
                AVL:$vl,
                !or(vti.SEW, nontemporal))>;

    def : Pat<(vti.Vector (int_epi_vload_nt_strided_mask
                           (vti.Vector vti.MInfo.vrclass:$merge), GPR:$rs1, GPR:$rs2,
                           (vti.Mask V0), AVL:$vl)),
              (!cast<Instruction>("PseudoEPIVLSE" # vti.SEW # "_V_" # MInfo.MX)
               $merge,
               $rs1, $rs2,
               (vti.Mask V0),
               AVL:$vl,
               !or(vti.SEW, nontemporal))>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(vti.Vector (int_epi_vload_nt_indexed_mask
                             (vti.Vector vti.MInfo.vrclass:$merge), GPR:$rs1,
                             (ivti.Vector ivti.MInfo.vrclass:$rs2), (vti.Mask V0),
                             AVL:$vl)),
                (!cast<Instruction>("PseudoEPIVLXE" # vti.SEW # "I" # ieew # "_V_" # MInfo.MX)
                 $merge,
                 $rs1, $rs2,
                 (vti.Mask V0),
                 AVL:$vl,
                 !or(vti.SEW, nontemporal))>;
    }

    // Extended loads.
    def : Pat<(vti.Vector (int_epi_vload_ext GPR:$rs1, GPR:$flags, AVL:$vl)),
              (!cast<Instruction>("PseudoEPIVLE" # vti.SEW # "_EXT_V_" # MInfo.MX)
               (vti.Vector (IMPLICIT_DEF)),
               $rs1,
               (vti.Mask zero_reg),
               GPR:$flags,
               AVL:$vl,
               vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_ext_strided GPR:$rs1, GPR:$rs2, GPR:$flags, AVL:$vl)),
              (!cast<Instruction>("PseudoEPIVLSE" # vti.SEW # "_EXT_V_" # MInfo.MX)
               (vti.Vector (IMPLICIT_DEF)),
               $rs1, $rs2,
               (vti.Mask zero_reg),
               GPR:$flags,
               AVL:$vl,
               vti.SEW)>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(vti.Vector (int_epi_vload_ext_indexed GPR:$rs1,
                             (ivti.Vector ivti.MInfo.vrclass:$rs2), GPR:$flags, AVL:$vl)),
                (!cast<Instruction>("PseudoEPIVLXE" # vti.SEW # "I" # ieew # "_EXT_V_" # MInfo.MX)
                 (vti.Vector (IMPLICIT_DEF)),
                 $rs1, $rs2,
                 (vti.Mask zero_reg),
                 GPR:$flags,
                 AVL:$vl,
                 vti.SEW)>;
    }

    def : Pat<(vti.Vector (int_epi_vload_ext_mask (vti.Vector vti.MInfo.vrclass:$merge),
                           GPR:$rs1, GPR:$flags, (vti.Mask V0), AVL:$vl)),
               (!cast<Instruction>("PseudoEPIVLE" # vti.SEW # "_EXT_V_" # MInfo.MX)
                $merge,
                $rs1,
                (vti.Mask V0),
                GPR:$flags,
                AVL:$vl,
                vti.SEW)>;

    def : Pat<(vti.Vector (int_epi_vload_ext_strided_mask
                           (vti.Vector vti.MInfo.vrclass:$merge), GPR:$rs1, GPR:$rs2,
                           GPR:$flags, (vti.Mask V0), AVL:$vl)),
              (!cast<Instruction>("PseudoEPIVLSE" # vti.SEW # "_EXT_V_" # MInfo.MX)
               $merge,
               $rs1, $rs2,
               (vti.Mask V0),
               GPR:$flags,
               AVL:$vl,
               vti.SEW)>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(vti.Vector (int_epi_vload_ext_indexed_mask
                             (vti.Vector vti.MInfo.vrclass:$merge), GPR:$rs1,
                             (ivti.Vector ivti.MInfo.vrclass:$rs2), GPR:$flags,
                             (vti.Mask V0), AVL:$vl)),
                (!cast<Instruction>("PseudoEPIVLXE" # vti.SEW # "I" # ieew # "_EXT_V_" # MInfo.MX)
                 $merge,
                 $rs1, $rs2,
                 (vti.Mask V0),
                 GPR:$flags,
                 AVL:$vl,
                 vti.SEW)>;
    }
  }
}

multiclass pat_store {
  foreach vti = EPIAllVectors in {
    defvar MInfo = vti.MInfo;

    def : Pat<(int_epi_vstore (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1,
               AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSE" # vti.SEW # "_V_" # MInfo.MX)
               $rs3, $rs1,
               (vti.Mask zero_reg),
               AVL:$vl,
               vti.SEW)>;

    def : Pat<(int_epi_vstore_strided
               (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1, GPR:$rs2, AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSSE" # vti.SEW # "_V_" # MInfo.MX)
               $rs3, $rs1, $rs2,
               (vti.Mask zero_reg),
               AVL:$vl,
               vti.SEW)>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(int_epi_vstore_indexed
                 (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1,
                 (ivti.Vector ivti.MInfo.vrclass:$rs2), AVL:$vl),
                 (!cast<Instruction>("PseudoEPIVSXE" # vti.SEW # "I" # ieew # "_V_" # MInfo.MX)
                 $rs3, $rs1, $rs2,
                 (vti.Mask zero_reg),
                 AVL:$vl,
                 vti.SEW)>;
    }

    def : Pat<(int_epi_vstore_mask
               (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1, (vti.Mask V0),
               AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSE" # vti.SEW # "_V_" # MInfo.MX)
               $rs3, $rs1,
               (vti.Mask V0),
               AVL:$vl,
               vti.SEW)>;

    def : Pat<(int_epi_vstore_strided_mask
               (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1, GPR:$rs2,
               (vti.Mask V0), AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSSE" # vti.SEW # "_V_" # MInfo.MX)
               $rs3, $rs1, $rs2,
               (vti.Mask V0),
               AVL:$vl,
               vti.SEW)>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(int_epi_vstore_indexed_mask
                 (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1,
                 (ivti.Vector ivti.MInfo.vrclass:$rs2), (vti.Mask V0), AVL:$vl),
                 (!cast<Instruction>("PseudoEPIVSXE" # vti.SEW # "I" # ieew # "_V_" # MInfo.MX)
                 $rs3, $rs1, $rs2,
                 (vti.Mask V0),
                 AVL:$vl,
                 vti.SEW)>;
     }

    defvar nontemporal = !shl(1, 9);
    def : Pat<(int_epi_vstore_nt (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1,
               AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSE" # vti.SEW # "_V_" # MInfo.MX)
               $rs3, $rs1,
               (vti.Mask zero_reg),
               AVL:$vl,
               !or(vti.SEW, nontemporal))>;

    def : Pat<(int_epi_vstore_nt_strided
               (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1, GPR:$rs2, AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSSE" # vti.SEW # "_V_" # MInfo.MX)
               $rs3, $rs1, $rs2,
               (vti.Mask zero_reg),
               AVL:$vl,
               !or(vti.SEW, nontemporal))>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(int_epi_vstore_nt_indexed
                 (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1,
                 (ivti.Vector ivti.MInfo.vrclass:$rs2), AVL:$vl),
                 (!cast<Instruction>("PseudoEPIVSXE" # vti.SEW # "I" # ieew # "_V_" # MInfo.MX)
                 $rs3, $rs1, $rs2,
                 (vti.Mask zero_reg),
                 AVL:$vl,
                 !or(vti.SEW, nontemporal))>;
    }

    def : Pat<(int_epi_vstore_nt_mask
               (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1, (vti.Mask V0),
               AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSE" # vti.SEW # "_V_" # MInfo.MX)
               $rs3, $rs1,
               (vti.Mask V0),
               AVL:$vl,
               !or(vti.SEW, nontemporal))>;

    def : Pat<(int_epi_vstore_nt_strided_mask
               (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1, GPR:$rs2,
               (vti.Mask V0), AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSSE" # vti.SEW # "_V_" # MInfo.MX)
               $rs3, $rs1, $rs2,
               (vti.Mask V0),
               AVL:$vl,
               !or(vti.SEW, nontemporal))>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(int_epi_vstore_nt_indexed_mask
                 (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1,
                 (ivti.Vector ivti.MInfo.vrclass:$rs2), (vti.Mask V0), AVL:$vl),
                 (!cast<Instruction>("PseudoEPIVSXE" # vti.SEW # "I" # ieew # "_V_" # MInfo.MX)
                 $rs3, $rs1, $rs2,
                 (vti.Mask V0),
                 AVL:$vl,
                 !or(vti.SEW, nontemporal))>;
    }

    // Extended stores.
    def : Pat<(int_epi_vstore_ext (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1,
               GPR:$flags, AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSE" # vti.SEW # "_EXT_V_" # MInfo.MX)
               $rs3, $rs1,
               (vti.Mask zero_reg),
               GPR:$flags,
               AVL:$vl,
               vti.SEW)>;

    def : Pat<(int_epi_vstore_ext_strided
               (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1, GPR:$rs2, GPR:$flags, AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSSE" # vti.SEW # "_EXT_V_" # MInfo.MX)
               $rs3, $rs1, $rs2,
               (vti.Mask zero_reg),
               GPR:$flags,
               AVL:$vl,
               vti.SEW)>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(int_epi_vstore_ext_indexed
                 (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1,
                 (ivti.Vector ivti.MInfo.vrclass:$rs2), GPR:$flags, AVL:$vl),
                 (!cast<Instruction>("PseudoEPIVSXE" # vti.SEW # "I" # ieew # "_EXT_V_" # MInfo.MX)
                 $rs3, $rs1, $rs2,
                 (vti.Mask zero_reg),
                 GPR:$flags,
                 AVL:$vl,
                 vti.SEW)>;
    }

    def : Pat<(int_epi_vstore_ext_mask
               (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1, GPR:$flags,
               (vti.Mask V0), AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSE" # vti.SEW # "_EXT_V_" # MInfo.MX)
               $rs3, $rs1,
               (vti.Mask V0),
               GPR:$flags,
               AVL:$vl,
               vti.SEW)>;

    def : Pat<(int_epi_vstore_ext_strided_mask
               (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1, GPR:$rs2, GPR:$flags,
               (vti.Mask V0), AVL:$vl),
               (!cast<Instruction>("PseudoEPIVSSE" # vti.SEW # "_EXT_V_" # MInfo.MX)
               $rs3, $rs1, $rs2,
               (vti.Mask V0),
               GPR:$flags,
               AVL:$vl,
               vti.SEW)>;

    foreach ieew = AllowedEEW<MInfo.value, /* sew */ vti.SEW>.allowed_sew in {
      defvar ivti = GetIntVectorTypeInfo<vti, ieew>.Vti;
      def : Pat<(int_epi_vstore_ext_indexed_mask
                 (vti.Vector vti.MInfo.vrclass:$rs3), GPR:$rs1,
                 (ivti.Vector ivti.MInfo.vrclass:$rs2), GPR:$flags,
                 (vti.Mask V0), AVL:$vl),
                 (!cast<Instruction>("PseudoEPIVSXE" # vti.SEW # "I" # ieew # "_EXT_V_" # MInfo.MX)
                 $rs3, $rs1, $rs2,
                 (vti.Mask V0),
                 GPR:$flags,
                 AVL:$vl,
                 vti.SEW)>;
    }
  }
}

let Predicates = [HasVInstructions] in {

def : Pat<(nxv1f64 (masked_ld GPR:$addr, undef, (nxv1i1 V0), (nxv1f64 VR:$merge))),
          (PseudoEPIVLE64_V_M1
             (nxv1f64 VR:$merge),
             GPR:$addr,
             (nxv1i1 V0),
             /* vl */ X0,
             64)>;

def : Pat<(masked_st (nxv1f64 VR:$value), GPR:$addr, undef, (nxv1i1 V0)),
          (PseudoEPIVSE64_V_M1
              VR:$value, GPR:$addr, (nxv1i1 V0), /* vl */ X0, 64)>;

def : Pat<(nxv1i1 (setule (nxv1i64 VR:$rs1), (nxv1i64 VR:$rs2))),
          (PseudoEPIVMSLEU_VV_M1 (nxv1i64 (IMPLICIT_DEF)), VR:$rs1, VR:$rs2,
           (nxv1i1 zero_reg), /* vl */ X0, 64)>;

defm "" : pat_load;
defm "" : pat_store;

}

let Predicates = [HasVInstructions] in {

foreach tsize = 2 ... 8 in {

defvar TupleRegClass = !cast<RegisterClass>("VRN"#tsize#"M1");

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, usesCustomInserter = 1 in
def PseudoEPIImplicitVRN#tsize#M1 : Pseudo<(outs TupleRegClass:$rd), (ins), []>;

foreach vti = NoGroupVectors in {
  defvar ivti = GetIntVectorTypeInfo<vti, vti.SEW>.Vti;

  def : Pat<(!cast<SDNode>("riscv_vlseg"#tsize) GPR:$rs1, AVL:$vl, (XLenVT vti.SEW)),
            (!cast<Instruction>("PseudoEPIVLSEG"#tsize#"E"#vti.SEW#"_V_M1")
             (!cast<Instruction>("PseudoEPIImplicitVRN"#tsize#"M1")),
             $rs1,
             (nxv1i1 zero_reg),
             AVL:$vl,
             vti.SEW)>;

  def : Pat<(!cast<SDNode>("riscv_vsseg"#tsize) TupleRegClass:$rs3, GPR:$rs1, AVL:$vl, (XLenVT vti.SEW)),
            (!cast<Instruction>("PseudoEPIVSSEG"#tsize#"E"#vti.SEW#"_V_M1")
             $rs3, $rs1,
             (nxv1i1 zero_reg),
             AVL:$vl,
             vti.SEW)>;

  def : Pat<(!cast<SDNode>("riscv_vlsseg"#tsize) GPR:$rs1, GPR:$rs2, AVL:$vl, (XLenVT vti.SEW)),
            (!cast<Instruction>("PseudoEPIVLSSEG"#tsize#"E"#vti.SEW#"_V_M1")
             (!cast<Instruction>("PseudoEPIImplicitVRN"#tsize#"M1")),
             $rs1, $rs2,
             (nxv1i1 zero_reg),
             AVL:$vl,
             vti.SEW)>;

  def : Pat<(!cast<SDNode>("riscv_vssseg"#tsize) TupleRegClass:$rs3, GPR:$rs1, GPR:$rs2, AVL:$vl, (XLenVT vti.SEW)),
            (!cast<Instruction>("PseudoEPIVSSSEG"#tsize#"E"#vti.SEW#"_V_M1")
             $rs3, $rs1, $rs2,
             (nxv1i1 zero_reg),
             AVL:$vl,
             vti.SEW)>;

  def : Pat<(!cast<SDNode>("riscv_vlxseg"#tsize) GPR:$rs1,
               (ivti.Vector vti.MInfo.vrclass:$rs2), AVL:$vl, (XLenVT vti.SEW)),
            (!cast<Instruction>("PseudoEPIVLXSEG"#tsize#"EI"#vti.SEW#"_V_M1")
             (!cast<Instruction>("PseudoEPIImplicitVRN"#tsize#"M1")),
             $rs1, $rs2,
             (nxv1i1 zero_reg),
             AVL:$vl,
             vti.SEW)>;

  def : Pat<(!cast<SDNode>("riscv_vsxseg"#tsize) TupleRegClass:$rs3, GPR:$rs1,
               (ivti.Vector vti.MInfo.vrclass:$rs2), AVL:$vl, (XLenVT vti.SEW)),
            (!cast<Instruction>("PseudoEPIVSXSEG"#tsize#"EI"#vti.SEW#"_V_M1")
             $rs3, $rs1, $rs2,
             (nxv1i1 zero_reg),
             AVL:$vl,
             vti.SEW)>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, usesCustomInserter = 1 in

def PseudoEPIVBuildVRN#tsize#M1 :
     Pseudo<(outs TupleRegClass:$rd),
               !dag(ins,
                 !foreach(i, Iota<tsize>.Value, VR),
                 !foreach(i, Iota<tsize>.Value, "rs"#i)),
               []>;

} // tsize

} // [HasVInstructions]

//===----------------------------------------------------------------------===//
// EPI custom instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasVInstructions] in {

let VLMul = 0, hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

defm PseudoEPIVZIP2_VV : pseudo_binary<VRN2M1, VR, VR, V_M1, "">;
defm PseudoEPIVUNZIP2_VV : pseudo_binary<VRN2M1, VR, VR, V_M1, "">;
defm PseudoEPIVTRN_VV : pseudo_binary<VRN2M1, VR, VR, V_M1, "">;

}

foreach vti = NoGroupVectors in {

    def : Pat<(riscv_vzip2 (vti.Vector VR:$rs1), (vti.Vector VR:$rs2),
                           (i64 AVL:$vl)),
              (PseudoEPIVZIP2_VV_M1
               (PseudoEPIImplicitVRN2M1), vti.MInfo.vrclass:$rs1, vti.MInfo.vrclass:$rs2,
               (vti.Mask zero_reg), AVL:$vl, vti.SEW)>;

    def : Pat<(riscv_vunzip2 (vti.Vector VR:$rs1), (vti.Vector VR:$rs2),
                             (i64 AVL:$vl)),
              (PseudoEPIVUNZIP2_VV_M1
               (PseudoEPIImplicitVRN2M1), vti.MInfo.vrclass:$rs1, vti.MInfo.vrclass:$rs2,
               (vti.Mask zero_reg), AVL:$vl, vti.SEW)>;

    def : Pat<(riscv_vtrn (vti.Vector VR:$rs1), (vti.Vector VR:$rs2),
                          (i64 AVL:$vl)),
              (PseudoEPIVTRN_VV_M1
               (PseudoEPIImplicitVRN2M1), vti.MInfo.vrclass:$rs1, vti.MInfo.vrclass:$rs2,
               (vti.Mask zero_reg), AVL:$vl, vti.SEW)>;
}

}
