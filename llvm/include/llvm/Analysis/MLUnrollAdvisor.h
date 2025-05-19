//===- MLUnrollAdvisor.h - ML - based UnrollAdvisor factories ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MLUNROLLADVISOR_H
#define LLVM_ANALYSIS_MLUNROLLADVISOR_H

#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/UnrollAdvisor.h"
#include "llvm/IR/PassManager.h"

#include <map>
#include <memory>
#include <optional>

namespace llvm {
class DiagnosticInfoOptimizationBase;
class Module;
class MLUnrollAdvice;
class ProfileSummaryInfo;

class MLUnrollAdvisor : public UnrollAdvisor {
public:
  MLUnrollAdvisor(LLVMContext &Ctx) {}

  virtual ~MLUnrollAdvisor() = default;
  void onSuccessfulInlining(const MLUnrollAdvice &Advice,
                            bool CalleeWasDeleted) {}
  const MLModelRunner &getModelRunner() const { return *ModelRunner; }

protected:
  std::unique_ptr<UnrollAdvice> getAdviceImpl(UnrollAdviceInfo UAI) override;

  std::unique_ptr<MLModelRunner> ModelRunner;
  std::function<bool(CallBase &)> GetDefaultAdvice;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_MLUNROLLADVISOR_H
