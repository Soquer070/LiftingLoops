; RUN: llc -mtriple riscv64-unknown-linux-gnu -mattr=+m,+f,+d,+a,+c,+zepi < %s | FileCheck %s --check-prefix=EPI
; RUN: llc -mtriple riscv64-unknown-linux-gnu -mattr=+m,+f,+d,+a,+c < %s | FileCheck %s --check-prefix=RV64GC

; EPI: .attribute	5, "rv64i
; EPI-NOT: _z

; RV64GC: .attribute	5, "rv64i{{.*}}_zicsr

; ModuleID = 't.c'
define dso_local void @foo() {
entry:
  ret void
}
