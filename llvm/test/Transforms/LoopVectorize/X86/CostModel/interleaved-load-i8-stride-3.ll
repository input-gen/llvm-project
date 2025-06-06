; NOTE: Assertions have been autogenerated by utils/update_analyze_test_checks.py UTC_ARGS: --filter "LV: Found an estimated cost of [0-9]+ for VF [0-9]+ For instruction:\s*%v0 = load i8, ptr %in0"
; RUN: opt -passes=loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+sse2 --debug-only=loop-vectorize --disable-output < %s 2>&1 | FileCheck %s --check-prefix=SSE2
; RUN: opt -passes=loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx  --debug-only=loop-vectorize --disable-output < %s 2>&1 | FileCheck %s --check-prefix=AVX1
; RUN: opt -passes=loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx2 --debug-only=loop-vectorize --disable-output < %s 2>&1 | FileCheck %s --check-prefix=AVX2
; RUN: opt -passes=loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx512vl --debug-only=loop-vectorize --disable-output < %s 2>&1 | FileCheck %s --check-prefix=AVX512DQ
; RUN: opt -passes=loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx512vl,+avx512bw --debug-only=loop-vectorize --disable-output < %s 2>&1 | FileCheck %s --check-prefix=AVX512BW
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = global [1024 x i8] zeroinitializer, align 128
@B = global [1024 x i8] zeroinitializer, align 128

define void @test() {
; SSE2-LABEL: 'test'
; SSE2:  LV: Found an estimated cost of 1 for VF 1 For instruction: %v0 = load i8, ptr %in0, align 1
; SSE2:  Cost of 24 for VF 2: INTERLEAVE-GROUP with factor 3 at %v0
; SSE2:  Cost of 50 for VF 4: INTERLEAVE-GROUP with factor 3 at %v0
; SSE2:  Cost of 93 for VF 8: INTERLEAVE-GROUP with factor 3 at %v0
; SSE2:  Cost of 189 for VF 16: INTERLEAVE-GROUP with factor 3 at %v0
;
; AVX1-LABEL: 'test'
; AVX1:  LV: Found an estimated cost of 1 for VF 1 For instruction: %v0 = load i8, ptr %in0, align 1
; AVX1:  Cost of 16 for VF 2: INTERLEAVE-GROUP with factor 3 at %v0
; AVX1:  Cost of 27 for VF 4: INTERLEAVE-GROUP with factor 3 at %v0
; AVX1:  Cost of 52 for VF 8: INTERLEAVE-GROUP with factor 3 at %v0
; AVX1:  Cost of 99 for VF 16: INTERLEAVE-GROUP with factor 3 at %v0
; AVX1:  Cost of 201 for VF 32: INTERLEAVE-GROUP with factor 3 at %v0
;
; AVX2-LABEL: 'test'
; AVX2:  LV: Found an estimated cost of 1 for VF 1 For instruction: %v0 = load i8, ptr %in0, align 1
; AVX2:  Cost of 7 for VF 2: INTERLEAVE-GROUP with factor 3 at %v0
; AVX2:  Cost of 6 for VF 4: INTERLEAVE-GROUP with factor 3 at %v0
; AVX2:  Cost of 9 for VF 8: INTERLEAVE-GROUP with factor 3 at %v0
; AVX2:  Cost of 13 for VF 16: INTERLEAVE-GROUP with factor 3 at %v0
; AVX2:  Cost of 17 for VF 32: INTERLEAVE-GROUP with factor 3 at %v0
;
; AVX512DQ-LABEL: 'test'
; AVX512DQ:  LV: Found an estimated cost of 1 for VF 1 For instruction: %v0 = load i8, ptr %in0, align 1
; AVX512DQ:  Cost of 7 for VF 2: INTERLEAVE-GROUP with factor 3 at %v0
; AVX512DQ:  Cost of 6 for VF 4: INTERLEAVE-GROUP with factor 3 at %v0
; AVX512DQ:  Cost of 9 for VF 8: INTERLEAVE-GROUP with factor 3 at %v0
; AVX512DQ:  Cost of 14 for VF 16: INTERLEAVE-GROUP with factor 3 at %v0
; AVX512DQ:  Cost of 16 for VF 32: INTERLEAVE-GROUP with factor 3 at %v0
; AVX512DQ:  Cost of 405 for VF 64: INTERLEAVE-GROUP with factor 3 at %v0
;
; AVX512BW-LABEL: 'test'
; AVX512BW:  LV: Found an estimated cost of 1 for VF 1 For instruction: %v0 = load i8, ptr %in0, align 1
; AVX512BW:  Cost of 4 for VF 2: INTERLEAVE-GROUP with factor 3 at %v0
; AVX512BW:  Cost of 4 for VF 4: INTERLEAVE-GROUP with factor 3 at %v0
; AVX512BW:  Cost of 13 for VF 8: INTERLEAVE-GROUP with factor 3 at %v0
; AVX512BW:  Cost of 13 for VF 16: INTERLEAVE-GROUP with factor 3 at %v0
; AVX512BW:  Cost of 16 for VF 32: INTERLEAVE-GROUP with factor 3 at %v0
; AVX512BW:  Cost of 25 for VF 64: INTERLEAVE-GROUP with factor 3 at %v0
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]

  %iv.0 = add nuw nsw i64 %iv, 0
  %iv.1 = add nuw nsw i64 %iv, 1
  %iv.2 = add nuw nsw i64 %iv, 2

  %in0 = getelementptr inbounds [1024 x i8], ptr @A, i64 0, i64 %iv.0
  %in1 = getelementptr inbounds [1024 x i8], ptr @A, i64 0, i64 %iv.1
  %in2 = getelementptr inbounds [1024 x i8], ptr @A, i64 0, i64 %iv.2

  %v0 = load i8, ptr %in0
  %v1 = load i8, ptr %in1
  %v2 = load i8, ptr %in2

  %reduce.add.0 = add i8 %v0, %v1
  %reduce.add.1 = add i8 %reduce.add.0, %v2

  %out = getelementptr inbounds [1024 x i8], ptr @B, i64 0, i64 %iv.0
  store i8 %reduce.add.1, ptr %out

  %iv.next = add nuw nsw i64 %iv.0, 3
  %cmp = icmp ult i64 %iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}
