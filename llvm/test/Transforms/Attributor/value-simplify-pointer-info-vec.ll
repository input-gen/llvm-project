; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --check-attributes --check-globals
; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal  -attributor-annotate-decl-cs  -S < %s | FileCheck %s --check-prefixes=CHECK,TUNIT
; RUN: opt -aa-pipeline=basic-aa -passes=attributor-cgscc -attributor-manifest-internal  -attributor-annotate-decl-cs -S < %s | FileCheck %s --check-prefixes=CHECK,CGSCC
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @vec_write_0() {
; CHECK: Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
; CHECK-LABEL: define {{[^@]+}}@vec_write_0
; CHECK-SAME: () #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:    ret i32 0
;
  %a = alloca <2 x i32>
  store <2 x i32> <i32 0, i32 0>, ptr %a
  %l1 = load i32, ptr %a
  %g = getelementptr i32, ptr %a, i64 1
  %l2 = load i32, ptr %g
  %add = add i32 %l1, %l2
  ret i32 %add
}

define i32 @vec_write_1() {
; CHECK: Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
; CHECK-LABEL: define {{[^@]+}}@vec_write_1
; CHECK-SAME: () #[[ATTR0]] {
; CHECK-NEXT:    ret i32 10
;
  %a = alloca <2 x i32>
  store <2 x i32> <i32 5, i32 5>, ptr %a
  %l1B = load i32, ptr %a
  %g = getelementptr i32, ptr %a, i64 1
  %l2B = load i32, ptr %g
  %add = add i32 %l1B, %l2B
  ret i32 %add
}

; TODO: We should support this.
define i32 @vec_write_2() {
; CHECK: Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
; CHECK-LABEL: define {{[^@]+}}@vec_write_2
; CHECK-SAME: () #[[ATTR0]] {
; CHECK-NEXT:    [[A:%.*]] = alloca <2 x i32>, align 8
; CHECK-NEXT:    store <2 x i32> <i32 3, i32 5>, ptr [[A]], align 8
; CHECK-NEXT:    [[L1:%.*]] = load i32, ptr [[A]], align 8
; CHECK-NEXT:    [[G:%.*]] = getelementptr i32, ptr [[A]], i64 1
; CHECK-NEXT:    [[L2:%.*]] = load i32, ptr [[G]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add i32 [[L1]], [[L2]]
; CHECK-NEXT:    ret i32 [[ADD]]
;
  %a = alloca <2 x i32>
  store <2 x i32> <i32 3, i32 5>, ptr %a
  %l1 = load i32, ptr %a
  %g = getelementptr i32, ptr %a, i64 1
  %l2 = load i32, ptr %g
  %add = add i32 %l1, %l2
  ret i32 %add
}
define i32 @vec_write_3() {
; CHECK: Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
; CHECK-LABEL: define {{[^@]+}}@vec_write_3
; CHECK-SAME: () #[[ATTR0]] {
; CHECK-NEXT:    [[A:%.*]] = alloca <4 x i32>, align 16
; CHECK-NEXT:    store <2 x i32> splat (i32 3), ptr [[A]], align 16
; CHECK-NEXT:    [[G:%.*]] = getelementptr i32, ptr [[A]], i64 1
; CHECK-NEXT:    store <2 x i32> splat (i32 5), ptr [[G]], align 8
; CHECK-NEXT:    [[J:%.*]] = getelementptr i32, ptr [[G]], i64 1
; CHECK-NEXT:    [[L2B:%.*]] = load i32, ptr [[G]], align 8
; CHECK-NEXT:    [[ADD:%.*]] = add i32 3, [[L2B]]
; CHECK-NEXT:    ret i32 [[ADD]]
;
  %a = alloca <4 x i32>
  store <2 x i32> <i32 3, i32 3>, ptr %a
  %g = getelementptr i32, ptr %a, i64 1
  store <2 x i32> <i32 5, i32 5>, ptr %g
  %j = getelementptr i32, ptr %g, i64 1
  store <2 x i32> <i32 7, i32 7>, ptr %j
  %l1B = load i32, ptr %a
  %l2B = load i32, ptr %g
  %add = add i32 %l1B, %l2B
  ret i32 %add
}
define i32 @vec_write_4() {
; CHECK: Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
; CHECK-LABEL: define {{[^@]+}}@vec_write_4
; CHECK-SAME: () #[[ATTR0]] {
; CHECK-NEXT:    ret i32 13
;
  %a = alloca <4 x i32>
  store i32 3, ptr %a
  %g = getelementptr i32, ptr %a, i64 1
  store <2 x i32> <i32 5, i32 5>, ptr %g
  %j = getelementptr i32, ptr %g, i64 1
  %l1B = load i32, ptr %a
  %l2B = load i32, ptr %g
  %l3B = load i32, ptr %j
  %add1 = add i32 %l1B, %l2B
  %add2 = add i32 %l3B, %add1
  ret i32 %add2
}
define i32 @vec_write_5(i32 %arg) {
; CHECK: Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
; CHECK-LABEL: define {{[^@]+}}@vec_write_5
; CHECK-SAME: (i32 [[ARG:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:    [[A1:%.*]] = alloca i8, i32 12, align 16
; CHECK-NEXT:    store i32 [[ARG]], ptr [[A1]], align 16
; CHECK-NEXT:    [[ADD1:%.*]] = add i32 [[ARG]], 5
; CHECK-NEXT:    [[ADD2:%.*]] = add i32 5, [[ADD1]]
; CHECK-NEXT:    ret i32 [[ADD2]]
;
  %a = alloca <4 x i32>
  store i32 %arg, ptr %a
  %g = getelementptr i32, ptr %a, i64 1
  store <2 x i32> <i32 5, i32 5>, ptr %g
  %j = getelementptr i32, ptr %g, i64 1
  %l1B5 = load i32, ptr %a
  %l2B5 = load i32, ptr %g
  %l3B5 = load i32, ptr %j
  %add1 = add i32 %l1B5, %l2B5
  %add2 = add i32 %l3B5, %add1
  ret i32 %add2
}
;.
; CHECK: attributes #[[ATTR0]] = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
;.
;; NOTE: These prefixes are unused and the list is autogenerated. Do not add tests below this line:
; CGSCC: {{.*}}
; TUNIT: {{.*}}
