//===-- UnrollLoopDevelopmentAdvisor.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The development advisor communicates over the channels in the following way:
//
// << is output
// >> is input
//
// << {"observation" : <id : int>}
// << feature tensor : UnrollFeatureMap
// << {"heuristic" : <id : int>}
// << heuristic result : int64_t (i.e. unroll factor)
// >> Model Output : UnrollDecisionSpec
// << {"action" : <id : int>}
// << action result : uint8_t
// >> should_instrument : uint8_t
// if should_instrument
//   >> start_callback_name : cstring
//   >> end_callback_name : cstring
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/InteractiveModelRunner.h"
#include "llvm/Analysis/LoopPropertiesAnalysis.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/Analysis/ReleaseModeModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Analysis/UnrollAdvisor.h"
#include "llvm/Analysis/UnrollModelFeatureMaps.h"
#include "llvm/Analysis/Utils/TrainingLogger.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>

#define DEBUG_TYPE "loop-unroll-development-advisor"
#define DBGS() llvm::dbgs() << "mlgo-loop-unroll: "

using namespace llvm;

static cl::opt<std::string> InteractiveChannelBaseName(
    "mlgo-loop-unroll-interactive-channel-base", cl::Hidden,
    cl::desc(
        "Base file path for the interactive mode. The incoming filename should "
        "have the name <name>.in, while the outgoing name should be "
        "<name>.out"));

using namespace llvm::mlgo;

namespace {

class UnrollInteractiveModelRunner : public InteractiveModelRunner {
public:
  using InteractiveModelRunner::InteractiveModelRunner;

  static bool classof(const MLModelRunner *R) {
    return R->getKind() == MLModelRunner::Kind::UnrollInteractive;
  }

  void logHeuristic(std::optional<unsigned> UnrollFactor) {
    int64_t ToLog;
    if (UnrollFactor) {
      ToLog = *UnrollFactor;
    } else {
      ToLog = -1;
    }
    LLVM_DEBUG(DBGS() << "Logging factor " << ToLog << "\n");
    Log->logCustom<int64_t>("heuristic", ToLog);
    Log->flush();
  }

  void logAction(uint8_t Unrolled) {
    LLVM_DEBUG(DBGS() << "Logging action " << (int)Unrolled << "\n");
    Log->logCustom<uint8_t>("action", Unrolled);
    Log->flush();
  }

  UnrollAdvice::InstrumentationInfo getInstrumentation() {
    bool ShouldInstrument = read<uint8_t>();
    LLVM_DEBUG(DBGS() << "ShouldInstrument " << ShouldInstrument << "\n");
    if (!ShouldInstrument)
      return std::nullopt;

    auto BeginName = readString();
    auto EndName = readString();

    LLVM_DEBUG(DBGS() << "Instrumentation: " << BeginName << " " << EndName
                      << "\n");

    return UnrollAdvice::InstrumentationNames{BeginName, EndName};
  }

  std::string readString() {
    std::vector<char> OutputBuffer;
    while (true) {
      char C;
      auto ReadOrErr = ::sys::fs::readNativeFile(
          sys::fs::convertFDToNativeFile(Inbound), {&C, 1});
      if (ReadOrErr.takeError()) {
        Ctx.emitError("Failed reading from inbound file");
        OutputBuffer.back() = '\0';
        break;
      } else if (*ReadOrErr == 1) {
        OutputBuffer.push_back(C);
        if (C == '\0')
          break;
        else
          continue;
      } else if (*ReadOrErr == 0) {
        continue;
      }
      llvm_unreachable("???");
    }
    return OutputBuffer.data();
  }

  template <typename T> T read() {
    char Buff[sizeof(T)];
    readRaw(Buff, sizeof(T));
    return *reinterpret_cast<T *>(Buff);
  }
  void readRaw(char *Buff, size_t N) {
    size_t InsPoint = 0;
    const size_t Limit = N;
    while (InsPoint < Limit) {
      auto ReadOrErr =
          ::sys::fs::readNativeFile(sys::fs::convertFDToNativeFile(Inbound),
                                    {Buff + InsPoint, N - InsPoint});
      if (ReadOrErr.takeError()) {
        Ctx.emitError("Failed reading from inbound file");
        break;
      }
      InsPoint += *ReadOrErr;
    }
  }
};

class DevelopmentUnrollAdvisor : public UnrollAdvisor {
public:
  DevelopmentUnrollAdvisor(LLVMContext &Ctx)
      : ModelRunner(std::make_unique<UnrollInteractiveModelRunner>(
            Ctx, mlgo::UnrollFeatureMap, mlgo::UnrollDecisionSpec,
            InteractiveChannelBaseName + ".out",
            InteractiveChannelBaseName + ".in")) {}
  ~DevelopmentUnrollAdvisor() {}

  UnrollAdvice::InstrumentationInfo onAction(const LoopUnrollResult &Result) {
    assert((int)Result > 0);
    getModelRunner()->logAction(static_cast<uint8_t>(Result));
    return getModelRunner()->getInstrumentation();
  }
  UnrollAdvice::InstrumentationInfo onNoAction() {
    getModelRunner()->logAction(0);
    return getModelRunner()->getInstrumentation();
  }

protected:
  std::unique_ptr<UnrollAdvice> getAdviceImpl(UnrollAdviceInfo UAI) override;

private:
  UnrollInteractiveModelRunner *getModelRunner() { return ModelRunner.get(); }
  std::unique_ptr<UnrollInteractiveModelRunner> ModelRunner;
};

class DevelopmentUnrollAdvice : public UnrollAdvice {
public:
  using UnrollAdvice::UnrollAdvice;
  UnrollAdvice::InstrumentationInfo
  recordUnrollingImpl(const LoopUnrollResult &Result) override {
    assert(Result != LoopUnrollResult::Unmodified);
    LLVM_DEBUG(DBGS() << "unrolled\n");
    return getAdvisor()->onAction(Result);
  }
  UnrollAdvice::InstrumentationInfo
  recordUnsuccessfulUnrollingImpl(const LoopUnrollResult &Result) override {
    assert(Result == LoopUnrollResult::Unmodified);
    assert((int)Result == 0);
    LLVM_DEBUG(DBGS() << "unsuccessful unroll\n");
    return getAdvisor()->onNoAction();
  }
  UnrollAdvice::InstrumentationInfo recordUnattemptedUnrollingImpl() override {
    LLVM_DEBUG(DBGS() << "unattempted unroll\n");
    return getAdvisor()->onNoAction();
  }

private:
  DevelopmentUnrollAdvisor *getAdvisor() const {
    return static_cast<DevelopmentUnrollAdvisor *>(Advisor);
  };
};

std::unique_ptr<UnrollAdvice>
DevelopmentUnrollAdvisor::getAdviceImpl(UnrollAdviceInfo UAI) {
  LoopPropertiesInfo LPI = LoopPropertiesInfo::get(
      UAI.L, UAI.LI, UAI.SE, UAI.TTI, UAI.TLI, UAI.AA, UAI.DT, UAI.AC);

#define SET(NAME, val)                                                         \
  *ModelRunner->getTensor<int64_t>(UnrollFeatureIndex::NAME) =                 \
      static_cast<int64_t>(val);
  SET(loop_size, UAI.UCE.getRolledLoopSize());
  SET(trip_count, UAI.TripCount);
#undef SET

#define MAP_UINT_UINT_PROPERTY(NAME, DEFAULT)                                  \
  {                                                                            \
    auto BINS = llvm::mlgo::NAME##BinsNum;                                     \
    uint64_t *Tensor =                                                         \
        ModelRunner->getTensor<uint64_t>(UnrollFeatureIndex::NAME);            \
    for (unsigned I = 0; I < (unsigned)BINS; I++)                              \
      Tensor[I] = 0;                                                           \
  }
#define MAP_UINT64_UINT64_PROPERTY(NAME, DEFAULT)                              \
  MAP_UINT_UINT_PROPERTY(NAME, DEFAULT)
#define BOOL_PROPERTY(NAME, DEFAULT)                                           \
  *ModelRunner->getTensor<uint64_t>(UnrollFeatureIndex::NAME) =                \
      static_cast<uint64_t>(LPI.NAME);
#define UINT64_PROPERTY(NAME, DEFAULT)                                         \
  *ModelRunner->getTensor<uint64_t>(UnrollFeatureIndex::NAME) =                \
      static_cast<uint64_t>(LPI.NAME);
#define STRING_PROPERTY(NAME, DEFAULT)
#define APINT_PROPERTY(NAME, DEFAULT)                                          \
  {                                                                            \
    int64_t ToAssign;                                                          \
    if (auto V = LPI.NAME.trySExtValue())                                      \
      ToAssign = *V;                                                           \
    else                                                                       \
      ToAssign = std::numeric_limits<int64_t>::max();                          \
    *ModelRunner->getTensor<int64_t>(UnrollFeatureIndex::NAME) = ToAssign;     \
  }
#define INSTCOST_PROPERTY(NAME, DEFAULT)                                       \
  {                                                                            \
    int64_t ToAssign;                                                          \
    if (auto V = LPI.NAME.getValue())                                          \
      ToAssign = *V;                                                           \
    else                                                                       \
      ToAssign = -1;                                                           \
    *ModelRunner->getTensor<int64_t>(UnrollFeatureIndex::NAME) = ToAssign;     \
  }
#include "llvm/Analysis/LoopProperties.def"
#undef INSTCOST_PROPERTY
#undef BOOL_PROPERTY
#undef UINT64_PROPERTY
#undef STRING_PROPERTY
#undef APINT_PROPERTY
#undef MAP_UINT_UINT_PROPERTY
#undef MAP_UINT64_UINT64_PROPERTY

#define ENUM_BINS(NAME)                                                        \
  {                                                                            \
    uint64_t *Tensor =                                                         \
        ModelRunner->getTensor<uint64_t>(UnrollFeatureIndex::NAME);            \
    auto BINS = llvm::mlgo::NAME##BinsNum;                                     \
    for (unsigned I = 0; I < (unsigned)BINS; I++) {                            \
      auto Found = LPI.NAME.find(I);                                           \
      if (Found == LPI.NAME.end()) {                                           \
        Tensor[I] = 0;                                                         \
      } else {                                                                 \
        Tensor[I] = Found->second;                                             \
      }                                                                        \
    }                                                                          \
  }
  ENUM_BINS(RecurranceInfos)
  ENUM_BINS(DependenceInfos)
#undef ENUM_BINS

#define EXACT_BINS(NAME)                                                       \
  {                                                                            \
    uint64_t *Tensor =                                                         \
        ModelRunner->getTensor<uint64_t>(UnrollFeatureIndex::NAME);            \
    unsigned LT = NAME##Bins.size();                                           \
    unsigned GT = NAME##Bins.size() + 1;                                       \
    assert(Tensor[LT] == 0);                                                   \
    assert(Tensor[GT] == 0);                                                   \
    for (auto [Key, Val] : LPI.NAME) {                                         \
      const int *Found = llvm::find(NAME##Bins, Key);                          \
      if (Found == NAME##Bins.end()) {                                         \
        if ((int)Key < NAME##Bins.back())                                      \
          Tensor[LT] += Val;                                                   \
        else                                                                   \
          Tensor[GT] += Val;                                                   \
      } else {                                                                 \
        auto I = std::distance(NAME##Bins.begin(), Found);                     \
        Tensor[I] = Val;                                                       \
      }                                                                        \
    }                                                                          \
  }
  EXACT_BINS(AccessSizes)
  EXACT_BINS(AccessAlignments)
#undef EXACT_BINS

#define INTERVAL_BINS(NAME)                                                    \
  {                                                                            \
    uint64_t *Tensor =                                                         \
        ModelRunner->getTensor<uint64_t>(UnrollFeatureIndex::NAME);            \
    unsigned GT = NAME##Intervals.size();                                      \
    assert(Tensor[GT] == 0);                                                   \
    for (auto [Key, Val] : LPI.NAME) {                                         \
      unsigned I = 0;                                                          \
      while (I < NAME##Intervals.size() && NAME##Intervals[I] < (int)Key)      \
        I++;                                                                   \
      Tensor[I] += Val;                                                        \
    }                                                                          \
  }
  INTERVAL_BINS(LoopBlocksizes)
  INTERVAL_BINS(InstructionCostsRecipThroughput)
  INTERVAL_BINS(InstructionCostsLatency)
  INTERVAL_BINS(InstructionCostsCodeSize)
  INTERVAL_BINS(PtrStrides)
  INTERVAL_BINS(SpacialReuseDistance)
#undef INTERVAL_BINS

  ModelRunner->logInput();

  std::optional<unsigned> DefaultHeuristic = shouldPartialUnroll(
      UAI.UCE.getRolledLoopSize(), UAI.TripCount, UAI.UCE, UAI.UP);
  getModelRunner()->logHeuristic(DefaultHeuristic);
  if (DefaultHeuristic)
    LLVM_DEBUG(DBGS() << "default heuristic says " << *DefaultHeuristic
                      << "\n");
  else
    LLVM_DEBUG(DBGS() << "default heuristic says no unrolling\n");

  UnrollDecisionTy UD = ModelRunner->getOutput<UnrollDecisionTy>();

  std::optional<unsigned> UnrollFactor;

  auto IsMinusOne = [](float F) { return -0.99 >= F && F >= -1.01; };
  // Handle some special cases useful for training and evaluation.
  if (llvm::all_of(
          llvm::make_range(UD.Out, UD.Out + UnrollModelOutputLength - 1),
          IsMinusOne)) {
    float Last = UD.Out[UnrollModelOutputLength - 1];
    if (IsMinusOne(Last)) {
      LLVM_DEBUG(DBGS() << "got advice 'no decision'\n");
      UnrollFactor = std::nullopt;
    } else {
      long long LargeFactor = std::llround(Last);
      assert(LargeFactor > MaxUnrollFactor);
      UnrollFactor = LargeFactor;
      LLVM_DEBUG(DBGS() << "got large advice factor " << *UnrollFactor << "\n");
    }
  } else {
    // The model gives us a speedup estimate for each unroll factor in
    // [0,MaxUnrollFactor] whose indices are offset by UnrollFactorOffset.
    float *MaxEl = std::max_element(UD.Out, UD.Out + UnrollModelOutputLength);

    // Only unroll if the biggest estimated speedup is greater than 1.0.
    if (*MaxEl > 1.0) {
      unsigned ArgMax = std::distance(UD.Out, MaxEl);
      UnrollFactor = ArgMax + UnrollFactorOffset;
      LLVM_DEBUG(DBGS() << "got advice factor " << *UnrollFactor << "\n");
    } else {
      // Returning std::nullopt means that we made no decision, whereas we want
      // to say "we decided not to unroll", which is 0.
      UnrollFactor = 0;
      LLVM_DEBUG(DBGS() << "got advice nounroll\n");
    }
  }

  return std::make_unique<DevelopmentUnrollAdvice>(this, UnrollFactor);
}

} // namespace

std::unique_ptr<UnrollAdvisor>
llvm::getDevelopmentModeUnrollAdvisor(LLVMContext &Ctx) {
  return std::make_unique<DevelopmentUnrollAdvisor>(Ctx);
}

// clang-format off
const std::vector<TensorSpec> llvm::mlgo::UnrollFeatureMap{

#define POPULATE_NAMES(DTYPE, SHAPE, NAME, __) \
  TensorSpec::createSpec<DTYPE>(#NAME, SHAPE),
  LOOP_UNROLL_FEATURE_ITERATOR(POPULATE_NAMES)
#undef POPULATE_NAMES

#define MAP_UINT_UINT_PROPERTY(NAME, DEFAULT) TensorSpec::createSpec<uint64_t>(#NAME, {(unsigned)NAME##BinsNum}),
#define MAP_UINT64_UINT64_PROPERTY(NAME, DEFAULT) TensorSpec::createSpec<uint64_t>(#NAME, {(unsigned)NAME##BinsNum}),
#define BOOL_PROPERTY(NAME, DEFAULT) TensorSpec::createSpec<uint64_t>(#NAME, {1}),
#define UINT64_PROPERTY(NAME, DEFAULT) TensorSpec::createSpec<uint64_t>(#NAME, {1}),
#define STRING_PROPERTY(NAME, DEFAULT)
#define APINT_PROPERTY(NAME, DEFAULT) TensorSpec::createSpec<int64_t>(#NAME, {1}),
#define INSTCOST_PROPERTY(NAME, DEFAULT) TensorSpec::createSpec<int64_t>(#NAME, {1}),
#include "llvm/Analysis/LoopProperties.def"
#undef INSTCOST_PROPERTY
#undef BOOL_PROPERTY
#undef UINT64_PROPERTY
#undef STRING_PROPERTY
#undef APINT_PROPERTY
#undef MAP_UINT_UINT_PROPERTY
#undef MAP_UINT64_UINT64_PROPERTY

};
// clang-format on

const char *const llvm::mlgo::UnrollDecisionName = "unrolling_decision";
const TensorSpec llvm::mlgo::UnrollDecisionSpec = TensorSpec::createSpec<float>(
    UnrollDecisionName, {UnrollModelOutputLength});
