; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --scrub-attributes --version 5
; RUN: opt < %s -passes=lightsan -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"


define noundef zeroext i1 @_Z15store_load_boolPb(ptr captures(none) noundef initializes((0, 1)) %A) {
; CHECK-LABEL: define noundef zeroext i1 @_Z15store_load_boolPb(
; CHECK-SAME: ptr noundef captures(none) initializes((0, 1)) [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP3:%.*]] = call ptr @__lightsan_post_base_pointer_info(ptr [[A]], i32 0)
; CHECK-NEXT:    [[TMP0:%.*]] = call ptr @__lightsan_pre_store(ptr [[A]], ptr [[TMP3]], i32 1)
; CHECK-NEXT:    store i8 1, ptr [[TMP0]], align 1
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 1
; CHECK-NEXT:    [[TMP1:%.*]] = call ptr @__lightsan_pre_load(ptr [[ARRAYIDX]], ptr [[TMP3]], i32 1)
; CHECK-NEXT:    [[TMP2:%.*]] = load i8, ptr [[TMP1]], align 1
; CHECK-NEXT:    [[LOADEDV:%.*]] = trunc nuw i8 [[TMP2]] to i1
; CHECK-NEXT:    ret i1 [[LOADEDV]]
;
entry:
  store i8 1, ptr %A, align 1
  %arrayidx = getelementptr inbounds nuw i8, ptr %A, i64 1
  %0 = load i8, ptr %arrayidx, align 1
  %loadedv = trunc nuw i8 %0 to i1
  ret i1 %loadedv
}


define noundef signext i8 @_Z15store_load_charPc(ptr captures(none) noundef initializes((0, 1)) %A) {
; CHECK-LABEL: define noundef signext i8 @_Z15store_load_charPc(
; CHECK-SAME: ptr noundef captures(none) initializes((0, 1)) [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP3:%.*]] = call ptr @__lightsan_post_base_pointer_info(ptr [[A]], i32 0)
; CHECK-NEXT:    [[TMP0:%.*]] = call ptr @__lightsan_pre_store(ptr [[A]], ptr [[TMP3]], i32 1)
; CHECK-NEXT:    store i8 1, ptr [[TMP0]], align 1
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 1
; CHECK-NEXT:    [[TMP1:%.*]] = call ptr @__lightsan_pre_load(ptr [[ARRAYIDX]], ptr [[TMP3]], i32 1)
; CHECK-NEXT:    [[TMP2:%.*]] = load i8, ptr [[TMP1]], align 1
; CHECK-NEXT:    ret i8 [[TMP2]]
;
entry:
  store i8 1, ptr %A, align 1
  %arrayidx = getelementptr inbounds nuw i8, ptr %A, i64 1
  %0 = load i8, ptr %arrayidx, align 1
  ret i8 %0
}


define noundef signext i16 @_Z16store_load_shortPs(ptr captures(none) noundef initializes((0, 2)) %A) {
; CHECK-LABEL: define noundef signext i16 @_Z16store_load_shortPs(
; CHECK-SAME: ptr noundef captures(none) initializes((0, 2)) [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP3:%.*]] = call ptr @__lightsan_post_base_pointer_info(ptr [[A]], i32 0)
; CHECK-NEXT:    [[TMP0:%.*]] = call ptr @__lightsan_pre_store(ptr [[A]], ptr [[TMP3]], i32 2)
; CHECK-NEXT:    store i16 2, ptr [[TMP0]], align 2
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 2
; CHECK-NEXT:    [[TMP1:%.*]] = call ptr @__lightsan_pre_load(ptr [[ARRAYIDX]], ptr [[TMP3]], i32 2)
; CHECK-NEXT:    [[TMP2:%.*]] = load i16, ptr [[TMP1]], align 2
; CHECK-NEXT:    ret i16 [[TMP2]]
;
entry:
  store i16 2, ptr %A, align 2
  %arrayidx = getelementptr inbounds nuw i8, ptr %A, i64 2
  %0 = load i16, ptr %arrayidx, align 2
  ret i16 %0
}


define noundef i32 @_Z14store_load_intPi(ptr captures(none) noundef initializes((0, 4)) %A) {
; CHECK-LABEL: define noundef i32 @_Z14store_load_intPi(
; CHECK-SAME: ptr noundef captures(none) initializes((0, 4)) [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP3:%.*]] = call ptr @__lightsan_post_base_pointer_info(ptr [[A]], i32 0)
; CHECK-NEXT:    [[TMP0:%.*]] = call ptr @__lightsan_pre_store(ptr [[A]], ptr [[TMP3]], i32 4)
; CHECK-NEXT:    store i32 3, ptr [[TMP0]], align 4
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 4
; CHECK-NEXT:    [[TMP1:%.*]] = call ptr @__lightsan_pre_load(ptr [[ARRAYIDX]], ptr [[TMP3]], i32 4)
; CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[TMP1]], align 4
; CHECK-NEXT:    ret i32 [[TMP2]]
;
entry:
  store i32 3, ptr %A, align 4
  %arrayidx = getelementptr inbounds nuw i8, ptr %A, i64 4
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}


define noundef i64 @_Z15store_load_longPl(ptr captures(none) noundef initializes((0, 8)) %A) {
; CHECK-LABEL: define noundef i64 @_Z15store_load_longPl(
; CHECK-SAME: ptr noundef captures(none) initializes((0, 8)) [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP3:%.*]] = call ptr @__lightsan_post_base_pointer_info(ptr [[A]], i32 0)
; CHECK-NEXT:    [[TMP0:%.*]] = call ptr @__lightsan_pre_store(ptr [[A]], ptr [[TMP3]], i32 8)
; CHECK-NEXT:    store i64 4, ptr [[TMP0]], align 8
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 8
; CHECK-NEXT:    [[TMP1:%.*]] = call ptr @__lightsan_pre_load(ptr [[ARRAYIDX]], ptr [[TMP3]], i32 8)
; CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr [[TMP1]], align 8
; CHECK-NEXT:    ret i64 [[TMP2]]
;
entry:
  store i64 4, ptr %A, align 8
  %arrayidx = getelementptr inbounds nuw i8, ptr %A, i64 8
  %0 = load i64, ptr %arrayidx, align 8
  ret i64 %0
}


define noundef i128 @_Z20store_load_long_longPx(ptr captures(none) noundef initializes((0, 16)) %A) {
; CHECK-LABEL: define noundef i128 @_Z20store_load_long_longPx(
; CHECK-SAME: ptr noundef captures(none) initializes((0, 16)) [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP0:%.*]] = call ptr @__lightsan_post_base_pointer_info(ptr [[A]], i32 0)
; CHECK-NEXT:    [[TMP1:%.*]] = call ptr @__lightsan_pre_store(ptr [[A]], ptr [[TMP0]], i32 16)
; CHECK-NEXT:    store i128 5, ptr [[TMP1]], align 8
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 16
; CHECK-NEXT:    [[TMP2:%.*]] = call ptr @__lightsan_pre_load(ptr [[ARRAYIDX]], ptr [[TMP0]], i32 16)
; CHECK-NEXT:    [[TMP3:%.*]] = load i128, ptr [[TMP2]], align 8
; CHECK-NEXT:    ret i128 [[TMP3]]
;
entry:
  store i128 5, ptr %A, align 8
  %arrayidx = getelementptr inbounds nuw i8, ptr %A, i64 16
  %0 = load i128, ptr %arrayidx, align 8
  ret i128 %0
}


define noundef float @_Z16store_load_floatPf(ptr captures(none) noundef initializes((0, 4)) %A) {
; CHECK-LABEL: define noundef float @_Z16store_load_floatPf(
; CHECK-SAME: ptr noundef captures(none) initializes((0, 4)) [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP3:%.*]] = call ptr @__lightsan_post_base_pointer_info(ptr [[A]], i32 0)
; CHECK-NEXT:    [[TMP0:%.*]] = call ptr @__lightsan_pre_store(ptr [[A]], ptr [[TMP3]], i32 4)
; CHECK-NEXT:    store float 6.000000e+00, ptr [[TMP0]], align 4
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 4
; CHECK-NEXT:    [[TMP1:%.*]] = call ptr @__lightsan_pre_load(ptr [[ARRAYIDX]], ptr [[TMP3]], i32 4)
; CHECK-NEXT:    [[TMP2:%.*]] = load float, ptr [[TMP1]], align 4
; CHECK-NEXT:    ret float [[TMP2]]
;
entry:
  store float 6.000000e+00, ptr %A, align 4
  %arrayidx = getelementptr inbounds nuw i8, ptr %A, i64 4
  %0 = load float, ptr %arrayidx, align 4
  ret float %0
}


define noundef double @_Z17store_load_doublePd(ptr captures(none) noundef initializes((0, 8)) %A) {
; CHECK-LABEL: define noundef double @_Z17store_load_doublePd(
; CHECK-SAME: ptr noundef captures(none) initializes((0, 8)) [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP3:%.*]] = call ptr @__lightsan_post_base_pointer_info(ptr [[A]], i32 0)
; CHECK-NEXT:    [[TMP0:%.*]] = call ptr @__lightsan_pre_store(ptr [[A]], ptr [[TMP3]], i32 8)
; CHECK-NEXT:    store double 7.000000e+00, ptr [[TMP0]], align 8
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 8
; CHECK-NEXT:    [[TMP1:%.*]] = call ptr @__lightsan_pre_load(ptr [[ARRAYIDX]], ptr [[TMP3]], i32 8)
; CHECK-NEXT:    [[TMP2:%.*]] = load double, ptr [[TMP1]], align 8
; CHECK-NEXT:    ret double [[TMP2]]
;
entry:
  store double 7.000000e+00, ptr %A, align 8
  %arrayidx = getelementptr inbounds nuw i8, ptr %A, i64 8
  %0 = load double, ptr %arrayidx, align 8
  ret double %0
}


define noundef x86_fp80 @_Z22store_load_long_doublePe(ptr captures(none) noundef initializes((0, 10)) %A) {
; CHECK-LABEL: define noundef x86_fp80 @_Z22store_load_long_doublePe(
; CHECK-SAME: ptr noundef captures(none) initializes((0, 10)) [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP2:%.*]] = call ptr @__lightsan_post_base_pointer_info(ptr [[A]], i32 0)
; CHECK-NEXT:    [[TMP1:%.*]] = call ptr @__lightsan_pre_store(ptr [[A]], ptr [[TMP2]], i32 10)
; CHECK-NEXT:    store x86_fp80 0xK40028000000000000000, ptr [[TMP1]], align 16
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw i8, ptr [[A]], i64 16
; CHECK-NEXT:    [[TMP0:%.*]] = call ptr @__lightsan_pre_load(ptr [[ARRAYIDX]], ptr [[TMP2]], i32 10)
; CHECK-NEXT:    [[TMP4:%.*]] = load x86_fp80, ptr [[TMP0]], align 16
; CHECK-NEXT:    ret x86_fp80 [[TMP4]]
;
entry:
  store x86_fp80 0xK40028000000000000000, ptr %A, align 16
  %arrayidx = getelementptr inbounds nuw i8, ptr %A, i64 16
  %0 = load x86_fp80, ptr %arrayidx, align 16
  ret x86_fp80 %0
}
