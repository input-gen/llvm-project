; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -p loop-unroll -unroll-full-max-count=0 -S %s | FileCheck %s

declare void @foo(i32)

define i32 @umin_unit_step() {
; CHECK-LABEL: define i32 @umin_unit_step() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[SUB:%.*]] = sub i32 1024, [[IV]]
; CHECK-NEXT:    [[MINMAX:%.*]] = call i32 @llvm.umin.i32(i32 [[SUB]], i32 1)
; CHECK-NEXT:    call void @foo(i32 [[MINMAX]])
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i32 [[IV]], 1
; CHECK-NEXT:    [[EC:%.*]] = icmp ne i32 [[IV_NEXT]], 1024
; CHECK-NEXT:    br i1 [[EC]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[MINMAX_LCSSA:%.*]] = phi i32 [ [[MINMAX]], %[[LOOP]] ]
; CHECK-NEXT:    ret i32 [[MINMAX_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %sub = sub i32 1024, %iv
  %minmax = call i32 @llvm.umin(i32 %sub, i32 1)
  call void @foo(i32 %minmax)
  %iv.next = add nuw nsw i32 %iv, 1
  %ec = icmp ne i32 %iv.next, 1024
  br i1 %ec, label %loop, label %exit

exit:
  %minmax.lcssa = phi i32 [ %minmax, %loop ]
  ret i32 %minmax.lcssa
}

define i32 @smin_unit_step() {
; CHECK-LABEL: define i32 @smin_unit_step() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[SUB:%.*]] = sub i32 1024, [[IV]]
; CHECK-NEXT:    [[MINMAX:%.*]] = call i32 @llvm.smin.i32(i32 [[SUB]], i32 1)
; CHECK-NEXT:    call void @foo(i32 [[MINMAX]])
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i32 [[IV]], 1
; CHECK-NEXT:    [[EC_PEEL:%.*]] = icmp ne i32 [[IV_NEXT]], 1024
; CHECK-NEXT:    br i1 [[EC_PEEL]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[MINMAX_LCSSA:%.*]] = phi i32 [ [[MINMAX]], %[[LOOP]] ]
; CHECK-NEXT:    ret i32 [[MINMAX_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %sub = sub i32 1024, %iv
  %minmax = call i32 @llvm.smin(i32 %sub, i32 1)
  call void @foo(i32 %minmax)
  %iv.next = add nuw nsw i32 %iv, 1
  %ec = icmp ne i32 %iv.next, 1024
  br i1 %ec, label %loop, label %exit

exit:
  %minmax.lcssa = phi i32 [ %minmax, %loop ]
  ret i32 %minmax.lcssa
}

define i32 @smax_unit_step() {
; CHECK-LABEL: define i32 @smax_unit_step() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[SUB:%.*]] = sub i32 1024, [[IV]]
; CHECK-NEXT:    [[MINMAX:%.*]] = call i32 @llvm.smax.i32(i32 [[SUB]], i32 1)
; CHECK-NEXT:    call void @foo(i32 [[MINMAX]])
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i32 [[IV]], 1
; CHECK-NEXT:    [[EC_PEEL:%.*]] = icmp ne i32 [[IV_NEXT]], 1024
; CHECK-NEXT:    br i1 [[EC_PEEL]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[MINMAX_LCSSA:%.*]] = phi i32 [ [[MINMAX]], %[[LOOP]] ]
; CHECK-NEXT:    ret i32 [[MINMAX_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %sub = sub i32 1024, %iv
  %minmax = call i32 @llvm.smax(i32 %sub, i32 1)
  call void @foo(i32 %minmax)
  %iv.next = add nuw nsw i32 %iv, 1
  %ec = icmp ne i32 %iv.next, 1024
  br i1 %ec, label %loop, label %exit

exit:
  %minmax.lcssa = phi i32 [ %minmax, %loop ]
  ret i32 %minmax.lcssa
}

define i32 @umax_unit_step() {
; CHECK-LABEL: define i32 @umax_unit_step() {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[IV_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[SUB:%.*]] = sub i32 1024, [[IV]]
; CHECK-NEXT:    [[MINMAX:%.*]] = call i32 @llvm.umax.i32(i32 [[SUB]], i32 1)
; CHECK-NEXT:    call void @foo(i32 [[MINMAX]])
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i32 [[IV]], 1
; CHECK-NEXT:    [[EC:%.*]] = icmp ne i32 [[IV_NEXT]], 1024
; CHECK-NEXT:    br i1 [[EC]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    [[MINMAX_LCSSA:%.*]] = phi i32 [ [[MINMAX]], %[[LOOP]] ]
; CHECK-NEXT:    ret i32 [[MINMAX_LCSSA]]
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %sub = sub i32 1024, %iv
  %minmax = call i32 @llvm.umax(i32 %sub, i32 1)
  call void @foo(i32 %minmax)
  %iv.next = add nuw nsw i32 %iv, 1
  %ec = icmp ne i32 %iv.next, 1024
  br i1 %ec, label %loop, label %exit

exit:
  %minmax.lcssa = phi i32 [ %minmax, %loop ]
  ret i32 %minmax.lcssa
}

