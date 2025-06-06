; RUN: mkdir -p %t
; RUN: rm %t/extracted.* || true
; RUN: llvm-extract-loops -S %s --output-prefix %t/extracted. --output-suffix .ll --pretty-print-json
; RUN: cat %t/extracted.0.ll.json | FileCheck %s

; CHECK:  "loop_trip_count": "constant",


define void @foo(i32* %A) {
for.preheader:
  br label %for.body
for.body:
  %i = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
  store i32 %i, i32* %arrayidx, align 4
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, 10
  br i1 %cmp, label %for.body, label %for.exit
for.exit:
  br label %for.end
for.end:
  ret void
}
