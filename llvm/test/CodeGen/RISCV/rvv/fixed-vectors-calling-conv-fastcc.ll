; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-min=128 -riscv-v-fixed-length-vector-lmul-max=8 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,LMULMAX8
; RUN: llc -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-min=128 -riscv-v-fixed-length-vector-lmul-max=4 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,LMULMAX4

define fastcc <4 x i8> @ret_v4i8(ptr %p) {
; CHECK-LABEL: ret_v4i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli zero, 4, e8, mf4, ta, ma
; CHECK-NEXT:    vle8.v v8, (a0)
; CHECK-NEXT:    ret
  %v = load <4 x i8>, ptr %p
  ret <4 x i8> %v
}

define fastcc <4 x i32> @ret_v4i32(ptr %p) {
; CHECK-LABEL: ret_v4i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli zero, 4, e32, m1, ta, ma
; CHECK-NEXT:    vle32.v v8, (a0)
; CHECK-NEXT:    ret
  %v = load <4 x i32>, ptr %p
  ret <4 x i32> %v
}

define fastcc <8 x i32> @ret_v8i32(ptr %p) {
; CHECK-LABEL: ret_v8i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli zero, 8, e32, m2, ta, ma
; CHECK-NEXT:    vle32.v v8, (a0)
; CHECK-NEXT:    ret
  %v = load <8 x i32>, ptr %p
  ret <8 x i32> %v
}

define fastcc <16 x i64> @ret_v16i64(ptr %p) {
; LMULMAX8-LABEL: ret_v16i64:
; LMULMAX8:       # %bb.0:
; LMULMAX8-NEXT:    vsetivli zero, 16, e64, m8, ta, ma
; LMULMAX8-NEXT:    vle64.v v8, (a0)
; LMULMAX8-NEXT:    ret
;
; LMULMAX4-LABEL: ret_v16i64:
; LMULMAX4:       # %bb.0:
; LMULMAX4-NEXT:    vsetivli zero, 8, e64, m4, ta, ma
; LMULMAX4-NEXT:    vle64.v v8, (a0)
; LMULMAX4-NEXT:    addi a0, a0, 64
; LMULMAX4-NEXT:    vle64.v v12, (a0)
; LMULMAX4-NEXT:    ret
  %v = load <16 x i64>, ptr %p
  ret <16 x i64> %v
}

define fastcc <8 x i1> @ret_mask_v8i1(ptr %p) {
; CHECK-LABEL: ret_mask_v8i1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli zero, 8, e8, mf2, ta, ma
; CHECK-NEXT:    vlm.v v0, (a0)
; CHECK-NEXT:    ret
  %v = load <8 x i1>, ptr %p
  ret <8 x i1> %v
}

define fastcc <32 x i1> @ret_mask_v32i1(ptr %p) {
; CHECK-LABEL: ret_mask_v32i1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a1, 32
; CHECK-NEXT:    vsetvli zero, a1, e8, m2, ta, ma
; CHECK-NEXT:    vlm.v v0, (a0)
; CHECK-NEXT:    ret
  %v = load <32 x i1>, ptr %p
  ret <32 x i1> %v
}

; Return the vector via registers v8-v23
define fastcc <64 x i32> @ret_split_v64i32(ptr %x) {
; LMULMAX8-LABEL: ret_split_v64i32:
; LMULMAX8:       # %bb.0:
; LMULMAX8-NEXT:    li a1, 32
; LMULMAX8-NEXT:    vsetvli zero, a1, e32, m8, ta, ma
; LMULMAX8-NEXT:    vle32.v v8, (a0)
; LMULMAX8-NEXT:    addi a0, a0, 128
; LMULMAX8-NEXT:    vle32.v v16, (a0)
; LMULMAX8-NEXT:    ret
;
; LMULMAX4-LABEL: ret_split_v64i32:
; LMULMAX4:       # %bb.0:
; LMULMAX4-NEXT:    vsetivli zero, 16, e32, m4, ta, ma
; LMULMAX4-NEXT:    vle32.v v8, (a0)
; LMULMAX4-NEXT:    addi a1, a0, 64
; LMULMAX4-NEXT:    vle32.v v12, (a1)
; LMULMAX4-NEXT:    addi a1, a0, 128
; LMULMAX4-NEXT:    vle32.v v16, (a1)
; LMULMAX4-NEXT:    addi a0, a0, 192
; LMULMAX4-NEXT:    vle32.v v20, (a0)
; LMULMAX4-NEXT:    ret
  %v = load <64 x i32>, ptr %x
  ret <64 x i32> %v
}

; Return the vector fully via the stack
define fastcc <128 x i32> @ret_split_v128i32(ptr %x) {
; LMULMAX8-LABEL: ret_split_v128i32:
; LMULMAX8:       # %bb.0:
; LMULMAX8-NEXT:    addi a2, a1, 128
; LMULMAX8-NEXT:    li a3, 32
; LMULMAX8-NEXT:    vsetvli zero, a3, e32, m8, ta, ma
; LMULMAX8-NEXT:    vle32.v v8, (a2)
; LMULMAX8-NEXT:    addi a2, a1, 256
; LMULMAX8-NEXT:    vle32.v v16, (a1)
; LMULMAX8-NEXT:    addi a1, a1, 384
; LMULMAX8-NEXT:    vle32.v v24, (a1)
; LMULMAX8-NEXT:    vle32.v v0, (a2)
; LMULMAX8-NEXT:    vse32.v v16, (a0)
; LMULMAX8-NEXT:    addi a1, a0, 384
; LMULMAX8-NEXT:    vse32.v v24, (a1)
; LMULMAX8-NEXT:    addi a1, a0, 256
; LMULMAX8-NEXT:    vse32.v v0, (a1)
; LMULMAX8-NEXT:    addi a0, a0, 128
; LMULMAX8-NEXT:    vse32.v v8, (a0)
; LMULMAX8-NEXT:    ret
;
; LMULMAX4-LABEL: ret_split_v128i32:
; LMULMAX4:       # %bb.0:
; LMULMAX4-NEXT:    addi a2, a1, 64
; LMULMAX4-NEXT:    vsetivli zero, 16, e32, m4, ta, ma
; LMULMAX4-NEXT:    vle32.v v8, (a2)
; LMULMAX4-NEXT:    addi a2, a1, 128
; LMULMAX4-NEXT:    vle32.v v12, (a2)
; LMULMAX4-NEXT:    addi a2, a1, 192
; LMULMAX4-NEXT:    vle32.v v16, (a2)
; LMULMAX4-NEXT:    addi a2, a1, 256
; LMULMAX4-NEXT:    vle32.v v20, (a2)
; LMULMAX4-NEXT:    addi a2, a1, 320
; LMULMAX4-NEXT:    vle32.v v24, (a2)
; LMULMAX4-NEXT:    addi a2, a1, 384
; LMULMAX4-NEXT:    vle32.v v28, (a1)
; LMULMAX4-NEXT:    addi a1, a1, 448
; LMULMAX4-NEXT:    vle32.v v0, (a1)
; LMULMAX4-NEXT:    vle32.v v4, (a2)
; LMULMAX4-NEXT:    vse32.v v28, (a0)
; LMULMAX4-NEXT:    addi a1, a0, 448
; LMULMAX4-NEXT:    vse32.v v0, (a1)
; LMULMAX4-NEXT:    addi a1, a0, 384
; LMULMAX4-NEXT:    vse32.v v4, (a1)
; LMULMAX4-NEXT:    addi a1, a0, 320
; LMULMAX4-NEXT:    vse32.v v24, (a1)
; LMULMAX4-NEXT:    addi a1, a0, 256
; LMULMAX4-NEXT:    vse32.v v20, (a1)
; LMULMAX4-NEXT:    addi a1, a0, 192
; LMULMAX4-NEXT:    vse32.v v16, (a1)
; LMULMAX4-NEXT:    addi a1, a0, 128
; LMULMAX4-NEXT:    vse32.v v12, (a1)
; LMULMAX4-NEXT:    addi a0, a0, 64
; LMULMAX4-NEXT:    vse32.v v8, (a0)
; LMULMAX4-NEXT:    ret
  %v = load <128 x i32>, ptr %x
  ret <128 x i32> %v
}

define fastcc <4 x i8> @ret_v8i8_param_v4i8(<4 x i8> %v) {
; CHECK-LABEL: ret_v8i8_param_v4i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli zero, 4, e8, mf4, ta, ma
; CHECK-NEXT:    vadd.vi v8, v8, 2
; CHECK-NEXT:    ret
  %r = add <4 x i8> %v, <i8 2, i8 2, i8 2, i8 2>
  ret <4 x i8> %r
}

define fastcc <4 x i8> @ret_v4i8_param_v4i8_v4i8(<4 x i8> %v, <4 x i8> %w) {
; CHECK-LABEL: ret_v4i8_param_v4i8_v4i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli zero, 4, e8, mf4, ta, ma
; CHECK-NEXT:    vadd.vv v8, v8, v9
; CHECK-NEXT:    ret
  %r = add <4 x i8> %v, %w
  ret <4 x i8> %r
}

define fastcc <4 x i64> @ret_v4i64_param_v4i64_v4i64(<4 x i64> %v, <4 x i64> %w) {
; CHECK-LABEL: ret_v4i64_param_v4i64_v4i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli zero, 4, e64, m2, ta, ma
; CHECK-NEXT:    vadd.vv v8, v8, v10
; CHECK-NEXT:    ret
  %r = add <4 x i64> %v, %w
  ret <4 x i64> %r
}

define fastcc <8 x i1> @ret_v8i1_param_v8i1_v8i1(<8 x i1> %v, <8 x i1> %w) {
; CHECK-LABEL: ret_v8i1_param_v8i1_v8i1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli zero, 8, e8, mf2, ta, ma
; CHECK-NEXT:    vmxor.mm v0, v0, v8
; CHECK-NEXT:    ret
  %r = xor <8 x i1> %v, %w
  ret <8 x i1> %r
}

define fastcc <32 x i1> @ret_v32i1_param_v32i1_v32i1(<32 x i1> %v, <32 x i1> %w) {
; CHECK-LABEL: ret_v32i1_param_v32i1_v32i1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a0, 32
; CHECK-NEXT:    vsetvli zero, a0, e8, m2, ta, ma
; CHECK-NEXT:    vmand.mm v0, v0, v8
; CHECK-NEXT:    ret
  %r = and <32 x i1> %v, %w
  ret <32 x i1> %r
}

define fastcc <32 x i32> @ret_v32i32_param_v32i32_v32i32_v32i32_i32(<32 x i32> %x, <32 x i32> %y, <32 x i32> %z, i32 %w) {
; LMULMAX8-LABEL: ret_v32i32_param_v32i32_v32i32_v32i32_i32:
; LMULMAX8:       # %bb.0:
; LMULMAX8-NEXT:    li a2, 32
; LMULMAX8-NEXT:    vsetvli zero, a2, e32, m8, ta, ma
; LMULMAX8-NEXT:    vle32.v v24, (a0)
; LMULMAX8-NEXT:    vadd.vv v8, v8, v16
; LMULMAX8-NEXT:    vadd.vv v8, v8, v24
; LMULMAX8-NEXT:    vadd.vx v8, v8, a1
; LMULMAX8-NEXT:    ret
;
; LMULMAX4-LABEL: ret_v32i32_param_v32i32_v32i32_v32i32_i32:
; LMULMAX4:       # %bb.0:
; LMULMAX4-NEXT:    vsetivli zero, 16, e32, m4, ta, ma
; LMULMAX4-NEXT:    addi a1, a0, 64
; LMULMAX4-NEXT:    vle32.v v24, (a1)
; LMULMAX4-NEXT:    vle32.v v28, (a0)
; LMULMAX4-NEXT:    vadd.vv v8, v8, v16
; LMULMAX4-NEXT:    vadd.vv v12, v12, v20
; LMULMAX4-NEXT:    vadd.vv v12, v12, v24
; LMULMAX4-NEXT:    vadd.vv v8, v8, v28
; LMULMAX4-NEXT:    vadd.vx v8, v8, a2
; LMULMAX4-NEXT:    vadd.vx v12, v12, a2
; LMULMAX4-NEXT:    ret
  %r = add <32 x i32> %x, %y
  %s = add <32 x i32> %r, %z
  %head = insertelement <32 x i32> poison, i32 %w, i32 0
  %splat = shufflevector <32 x i32> %head, <32 x i32> poison, <32 x i32> zeroinitializer
  %t = add <32 x i32> %s, %splat
  ret <32 x i32> %t
}

declare <32 x i32> @ext2(<32 x i32>, <32 x i32>, i32, i32)
declare <32 x i32> @ext3(<32 x i32>, <32 x i32>, <32 x i32>, i32, i32)

define fastcc <32 x i32> @ret_v32i32_call_v32i32_v32i32_i32(<32 x i32> %x, <32 x i32> %y, i32 %w) {
; LMULMAX8-LABEL: ret_v32i32_call_v32i32_v32i32_i32:
; LMULMAX8:       # %bb.0:
; LMULMAX8-NEXT:    addi sp, sp, -16
; LMULMAX8-NEXT:    .cfi_def_cfa_offset 16
; LMULMAX8-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; LMULMAX8-NEXT:    .cfi_offset ra, -8
; LMULMAX8-NEXT:    vmv8r.v v24, v8
; LMULMAX8-NEXT:    li a1, 2
; LMULMAX8-NEXT:    vmv8r.v v8, v16
; LMULMAX8-NEXT:    vmv8r.v v16, v24
; LMULMAX8-NEXT:    call ext2@plt
; LMULMAX8-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; LMULMAX8-NEXT:    addi sp, sp, 16
; LMULMAX8-NEXT:    ret
;
; LMULMAX4-LABEL: ret_v32i32_call_v32i32_v32i32_i32:
; LMULMAX4:       # %bb.0:
; LMULMAX4-NEXT:    addi sp, sp, -16
; LMULMAX4-NEXT:    .cfi_def_cfa_offset 16
; LMULMAX4-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; LMULMAX4-NEXT:    .cfi_offset ra, -8
; LMULMAX4-NEXT:    vmv4r.v v24, v12
; LMULMAX4-NEXT:    vmv4r.v v28, v8
; LMULMAX4-NEXT:    li a1, 2
; LMULMAX4-NEXT:    vmv4r.v v8, v16
; LMULMAX4-NEXT:    vmv4r.v v12, v20
; LMULMAX4-NEXT:    vmv4r.v v16, v28
; LMULMAX4-NEXT:    vmv4r.v v20, v24
; LMULMAX4-NEXT:    call ext2@plt
; LMULMAX4-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; LMULMAX4-NEXT:    addi sp, sp, 16
; LMULMAX4-NEXT:    ret
  %t = call fastcc <32 x i32> @ext2(<32 x i32> %y, <32 x i32> %x, i32 %w, i32 2)
  ret <32 x i32> %t
}

define fastcc <32 x i32> @ret_v32i32_call_v32i32_v32i32_v32i32_i32(<32 x i32> %x, <32 x i32> %y, <32 x i32> %z, i32 %w) {
; LMULMAX8-LABEL: ret_v32i32_call_v32i32_v32i32_v32i32_i32:
; LMULMAX8:       # %bb.0:
; LMULMAX8-NEXT:    addi sp, sp, -144
; LMULMAX8-NEXT:    .cfi_def_cfa_offset 144
; LMULMAX8-NEXT:    sd ra, 136(sp) # 8-byte Folded Spill
; LMULMAX8-NEXT:    .cfi_offset ra, -8
; LMULMAX8-NEXT:    li a2, 32
; LMULMAX8-NEXT:    vsetvli zero, a2, e32, m8, ta, ma
; LMULMAX8-NEXT:    vle32.v v24, (a0)
; LMULMAX8-NEXT:    mv a3, sp
; LMULMAX8-NEXT:    mv a0, sp
; LMULMAX8-NEXT:    li a2, 42
; LMULMAX8-NEXT:    vse32.v v8, (a3)
; LMULMAX8-NEXT:    vmv.v.v v8, v24
; LMULMAX8-NEXT:    call ext3@plt
; LMULMAX8-NEXT:    ld ra, 136(sp) # 8-byte Folded Reload
; LMULMAX8-NEXT:    addi sp, sp, 144
; LMULMAX8-NEXT:    ret
;
; LMULMAX4-LABEL: ret_v32i32_call_v32i32_v32i32_v32i32_i32:
; LMULMAX4:       # %bb.0:
; LMULMAX4-NEXT:    addi sp, sp, -144
; LMULMAX4-NEXT:    .cfi_def_cfa_offset 144
; LMULMAX4-NEXT:    sd ra, 136(sp) # 8-byte Folded Spill
; LMULMAX4-NEXT:    .cfi_offset ra, -8
; LMULMAX4-NEXT:    vsetivli zero, 16, e32, m4, ta, ma
; LMULMAX4-NEXT:    vle32.v v24, (a0)
; LMULMAX4-NEXT:    addi a0, a0, 64
; LMULMAX4-NEXT:    vle32.v v28, (a0)
; LMULMAX4-NEXT:    addi a0, sp, 64
; LMULMAX4-NEXT:    vse32.v v12, (a0)
; LMULMAX4-NEXT:    mv a1, sp
; LMULMAX4-NEXT:    mv a0, sp
; LMULMAX4-NEXT:    li a3, 42
; LMULMAX4-NEXT:    vse32.v v8, (a1)
; LMULMAX4-NEXT:    vmv.v.v v8, v24
; LMULMAX4-NEXT:    vmv.v.v v12, v28
; LMULMAX4-NEXT:    call ext3@plt
; LMULMAX4-NEXT:    ld ra, 136(sp) # 8-byte Folded Reload
; LMULMAX4-NEXT:    addi sp, sp, 144
; LMULMAX4-NEXT:    ret
  %t = call fastcc <32 x i32> @ext3(<32 x i32> %z, <32 x i32> %y, <32 x i32> %x, i32 %w, i32 42)
  ret <32 x i32> %t
}

; A test case where the normal calling convention would pass directly via the
; stack, but with fastcc can pass indirectly with the extra GPR registers
; allowed.
define fastcc <32 x i32> @vector_arg_indirect_stack(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, <32 x i32> %x, <32 x i32> %y, <32 x i32> %z, i32 %8) {
; LMULMAX8-LABEL: vector_arg_indirect_stack:
; LMULMAX8:       # %bb.0:
; LMULMAX8-NEXT:    li a0, 32
; LMULMAX8-NEXT:    vsetvli zero, a0, e32, m8, ta, ma
; LMULMAX8-NEXT:    vle32.v v16, (t2)
; LMULMAX8-NEXT:    vadd.vv v8, v8, v16
; LMULMAX8-NEXT:    ret
;
; LMULMAX4-LABEL: vector_arg_indirect_stack:
; LMULMAX4:       # %bb.0:
; LMULMAX4-NEXT:    addi a0, t2, 64
; LMULMAX4-NEXT:    vsetivli zero, 16, e32, m4, ta, ma
; LMULMAX4-NEXT:    vle32.v v16, (t2)
; LMULMAX4-NEXT:    vle32.v v20, (a0)
; LMULMAX4-NEXT:    vadd.vv v8, v8, v16
; LMULMAX4-NEXT:    vadd.vv v12, v12, v20
; LMULMAX4-NEXT:    ret
  %s = add <32 x i32> %x, %z
  ret <32 x i32> %s
}

; Calling the function above. Ensure we pass the arguments correctly.
define fastcc <32 x i32> @pass_vector_arg_indirect_stack(<32 x i32> %x, <32 x i32> %y, <32 x i32> %z) {
; LMULMAX8-LABEL: pass_vector_arg_indirect_stack:
; LMULMAX8:       # %bb.0:
; LMULMAX8-NEXT:    addi sp, sp, -144
; LMULMAX8-NEXT:    .cfi_def_cfa_offset 144
; LMULMAX8-NEXT:    sd ra, 136(sp) # 8-byte Folded Spill
; LMULMAX8-NEXT:    .cfi_offset ra, -8
; LMULMAX8-NEXT:    li a0, 32
; LMULMAX8-NEXT:    vsetvli zero, a0, e32, m8, ta, ma
; LMULMAX8-NEXT:    vmv.v.i v8, 0
; LMULMAX8-NEXT:    mv a0, sp
; LMULMAX8-NEXT:    li a1, 1
; LMULMAX8-NEXT:    li a2, 2
; LMULMAX8-NEXT:    li a3, 3
; LMULMAX8-NEXT:    li a4, 4
; LMULMAX8-NEXT:    li a5, 5
; LMULMAX8-NEXT:    li a6, 6
; LMULMAX8-NEXT:    li a7, 7
; LMULMAX8-NEXT:    mv t2, sp
; LMULMAX8-NEXT:    li t3, 8
; LMULMAX8-NEXT:    vse32.v v8, (a0)
; LMULMAX8-NEXT:    li a0, 0
; LMULMAX8-NEXT:    vmv.v.i v16, 0
; LMULMAX8-NEXT:    call vector_arg_indirect_stack@plt
; LMULMAX8-NEXT:    ld ra, 136(sp) # 8-byte Folded Reload
; LMULMAX8-NEXT:    addi sp, sp, 144
; LMULMAX8-NEXT:    ret
;
; LMULMAX4-LABEL: pass_vector_arg_indirect_stack:
; LMULMAX4:       # %bb.0:
; LMULMAX4-NEXT:    addi sp, sp, -144
; LMULMAX4-NEXT:    .cfi_def_cfa_offset 144
; LMULMAX4-NEXT:    sd ra, 136(sp) # 8-byte Folded Spill
; LMULMAX4-NEXT:    .cfi_offset ra, -8
; LMULMAX4-NEXT:    vsetivli zero, 16, e32, m4, ta, ma
; LMULMAX4-NEXT:    vmv.v.i v8, 0
; LMULMAX4-NEXT:    addi a0, sp, 64
; LMULMAX4-NEXT:    vse32.v v8, (a0)
; LMULMAX4-NEXT:    mv a0, sp
; LMULMAX4-NEXT:    li a1, 1
; LMULMAX4-NEXT:    li a2, 2
; LMULMAX4-NEXT:    li a3, 3
; LMULMAX4-NEXT:    li a4, 4
; LMULMAX4-NEXT:    li a5, 5
; LMULMAX4-NEXT:    li a6, 6
; LMULMAX4-NEXT:    li a7, 7
; LMULMAX4-NEXT:    mv t2, sp
; LMULMAX4-NEXT:    li t4, 8
; LMULMAX4-NEXT:    vse32.v v8, (a0)
; LMULMAX4-NEXT:    li a0, 0
; LMULMAX4-NEXT:    vmv.v.i v12, 0
; LMULMAX4-NEXT:    vmv.v.i v16, 0
; LMULMAX4-NEXT:    vmv.v.i v20, 0
; LMULMAX4-NEXT:    call vector_arg_indirect_stack@plt
; LMULMAX4-NEXT:    ld ra, 136(sp) # 8-byte Folded Reload
; LMULMAX4-NEXT:    addi sp, sp, 144
; LMULMAX4-NEXT:    ret
  %s = call fastcc <32 x i32> @vector_arg_indirect_stack(i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, <32 x i32> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> zeroinitializer, i32 8)
  ret <32 x i32> %s
}

; A pathological test case where even with fastcc we must use the stack for arguments %13 and %z
define fastcc <32 x i32> @vector_arg_direct_stack(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, <32 x i32> %x, <32 x i32> %y, <32 x i32> %z, i32 %last) {
; LMULMAX8-LABEL: vector_arg_direct_stack:
; LMULMAX8:       # %bb.0:
; LMULMAX8-NEXT:    li a0, 32
; LMULMAX8-NEXT:    addi a1, sp, 8
; LMULMAX8-NEXT:    vsetvli zero, a0, e32, m8, ta, ma
; LMULMAX8-NEXT:    vle32.v v24, (a1)
; LMULMAX8-NEXT:    vadd.vv v8, v8, v16
; LMULMAX8-NEXT:    vadd.vv v8, v8, v24
; LMULMAX8-NEXT:    ret
;
; LMULMAX4-LABEL: vector_arg_direct_stack:
; LMULMAX4:       # %bb.0:
; LMULMAX4-NEXT:    vsetivli zero, 16, e32, m4, ta, ma
; LMULMAX4-NEXT:    addi a0, sp, 8
; LMULMAX4-NEXT:    vle32.v v24, (a0)
; LMULMAX4-NEXT:    addi a0, sp, 72
; LMULMAX4-NEXT:    vle32.v v28, (a0)
; LMULMAX4-NEXT:    vadd.vv v12, v12, v20
; LMULMAX4-NEXT:    vadd.vv v8, v8, v16
; LMULMAX4-NEXT:    vadd.vv v8, v8, v24
; LMULMAX4-NEXT:    vadd.vv v12, v12, v28
; LMULMAX4-NEXT:    ret
  %s = add <32 x i32> %x, %y
  %t = add <32 x i32> %s, %z
  ret <32 x i32> %t
}

; Calling the function above. Ensure we pass the arguments correctly.
define fastcc <32 x i32> @pass_vector_arg_direct_stack(<32 x i32> %x, <32 x i32> %y, <32 x i32> %z) {
; LMULMAX8-LABEL: pass_vector_arg_direct_stack:
; LMULMAX8:       # %bb.0:
; LMULMAX8-NEXT:    addi sp, sp, -160
; LMULMAX8-NEXT:    .cfi_def_cfa_offset 160
; LMULMAX8-NEXT:    sd ra, 152(sp) # 8-byte Folded Spill
; LMULMAX8-NEXT:    .cfi_offset ra, -8
; LMULMAX8-NEXT:    li a0, 32
; LMULMAX8-NEXT:    vsetvli zero, a0, e32, m8, ta, ma
; LMULMAX8-NEXT:    vmv.v.i v8, 0
; LMULMAX8-NEXT:    addi a0, sp, 8
; LMULMAX8-NEXT:    vse32.v v8, (a0)
; LMULMAX8-NEXT:    li a0, 1
; LMULMAX8-NEXT:    sd a0, 136(sp)
; LMULMAX8-NEXT:    li a0, 13
; LMULMAX8-NEXT:    li a1, 1
; LMULMAX8-NEXT:    li a2, 2
; LMULMAX8-NEXT:    li a3, 3
; LMULMAX8-NEXT:    li a4, 4
; LMULMAX8-NEXT:    li a5, 5
; LMULMAX8-NEXT:    li a6, 6
; LMULMAX8-NEXT:    li a7, 7
; LMULMAX8-NEXT:    li t2, 8
; LMULMAX8-NEXT:    li t3, 9
; LMULMAX8-NEXT:    li t4, 10
; LMULMAX8-NEXT:    li t5, 11
; LMULMAX8-NEXT:    li t6, 12
; LMULMAX8-NEXT:    sd a0, 0(sp)
; LMULMAX8-NEXT:    li a0, 0
; LMULMAX8-NEXT:    vmv.v.i v16, 0
; LMULMAX8-NEXT:    call vector_arg_direct_stack@plt
; LMULMAX8-NEXT:    ld ra, 152(sp) # 8-byte Folded Reload
; LMULMAX8-NEXT:    addi sp, sp, 160
; LMULMAX8-NEXT:    ret
;
; LMULMAX4-LABEL: pass_vector_arg_direct_stack:
; LMULMAX4:       # %bb.0:
; LMULMAX4-NEXT:    addi sp, sp, -160
; LMULMAX4-NEXT:    .cfi_def_cfa_offset 160
; LMULMAX4-NEXT:    sd ra, 152(sp) # 8-byte Folded Spill
; LMULMAX4-NEXT:    .cfi_offset ra, -8
; LMULMAX4-NEXT:    li a0, 1
; LMULMAX4-NEXT:    sd a0, 136(sp)
; LMULMAX4-NEXT:    li a0, 13
; LMULMAX4-NEXT:    sd a0, 0(sp)
; LMULMAX4-NEXT:    vsetivli zero, 16, e32, m4, ta, ma
; LMULMAX4-NEXT:    vmv.v.i v8, 0
; LMULMAX4-NEXT:    addi a0, sp, 72
; LMULMAX4-NEXT:    vse32.v v8, (a0)
; LMULMAX4-NEXT:    addi a0, sp, 8
; LMULMAX4-NEXT:    li a1, 1
; LMULMAX4-NEXT:    li a2, 2
; LMULMAX4-NEXT:    li a3, 3
; LMULMAX4-NEXT:    li a4, 4
; LMULMAX4-NEXT:    li a5, 5
; LMULMAX4-NEXT:    li a6, 6
; LMULMAX4-NEXT:    li a7, 7
; LMULMAX4-NEXT:    li t2, 8
; LMULMAX4-NEXT:    li t3, 9
; LMULMAX4-NEXT:    li t4, 10
; LMULMAX4-NEXT:    li t5, 11
; LMULMAX4-NEXT:    li t6, 12
; LMULMAX4-NEXT:    vse32.v v8, (a0)
; LMULMAX4-NEXT:    li a0, 0
; LMULMAX4-NEXT:    vmv.v.i v12, 0
; LMULMAX4-NEXT:    vmv.v.i v16, 0
; LMULMAX4-NEXT:    vmv.v.i v20, 0
; LMULMAX4-NEXT:    call vector_arg_direct_stack@plt
; LMULMAX4-NEXT:    ld ra, 152(sp) # 8-byte Folded Reload
; LMULMAX4-NEXT:    addi sp, sp, 160
; LMULMAX4-NEXT:    ret
  %s = call fastcc <32 x i32> @vector_arg_direct_stack(i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, <32 x i32> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> zeroinitializer, i32 1)
  ret <32 x i32> %s
}

; A pathological test case where even with fastcc we must use the stack for
; mask argument %m2. %m1 is passed via v0.
define fastcc <4 x i1> @vector_mask_arg_direct_stack(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, <32 x i32> %x, <32 x i32> %y, <32 x i32> %z, <4 x i1> %m1, <4 x i1> %m2, i32 %last) {
; CHECK-LABEL: vector_mask_arg_direct_stack:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a0, sp, 136
; CHECK-NEXT:    vsetivli zero, 4, e8, mf4, ta, ma
; CHECK-NEXT:    vlm.v v8, (a0)
; CHECK-NEXT:    vmxor.mm v0, v0, v8
; CHECK-NEXT:    ret
  %r = xor <4 x i1> %m1, %m2
  ret <4 x i1> %r
}
