
#include "llvm/Analysis/MLUnrollAdvisor.h"
#include "llvm/Analysis/LoopPropertiesAnalysis.h"
#include "llvm/Analysis/ModelUnderTrainingRunner.h"
#include "llvm/Analysis/ReleaseModeModelRunner.h"
#include "llvm/Analysis/UnrollAdvisor.h"
#include "llvm/Analysis/UnrollModelFeatureMaps.h"

#define DEBUG_TYPE "loop-unroll-ml-advisor"
#define DBGS() llvm::dbgs() << "mlgo-loop-unroll: "

using namespace llvm;
using namespace llvm::mlgo;

std::unique_ptr<UnrollAdvisor>
llvm::getReleaseModeUnrollAdvisor(LLVMContext &Ctx) {
  char *ModelPathEnv = getenv("LLVM_TFLITE_UNROLLMODEL_PATH");
  if (!ModelPathEnv)
    return nullptr;
  std::unique_ptr<MLModelRunner> AOTRunner;
  AOTRunner = ModelUnderTrainingRunner::createAndEnsureValid(
      Ctx, ModelPathEnv, UnrollDecisionName, UnrollFeatureMap);
  return std::make_unique<MLUnrollAdvisor>(Ctx);
}

std::unique_ptr<UnrollAdvice>
MLUnrollAdvisor::getAdviceImpl(UnrollAdviceInfo UAI) {
  // TODO need to pass the rest of the params, see if we can get them in the
  // unroll pass
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

  UnrollDecisionTy UD = ModelRunner->evaluate<UnrollDecisionTy>();

  // The model gives us a speedup estimate for each unroll factor in
  // [2,MaxUnrollFactor] whose indices are offset by UnrollFactorOffset.
  float *MaxEl = std::max_element(UD.Out, UD.Out + UnrollModelOutputLength);

  // Only unroll if the biggest estimated speedup is greater than 1.0.
  std::optional<unsigned> UnrollFactor;
  if (*MaxEl > 1.0) {
    unsigned ArgMax = std::distance(UD.Out, MaxEl);
    UnrollFactor = ArgMax + UnrollFactorOffset;
    LLVM_DEBUG(DBGS() << "got advice factor " << *UnrollFactor << "\n");
  } else {
    // Returning std::nullopt means that we made no decision, i.e. we delegated
    // the decision to later handlers. Thus, we need to return 0 meaning "we
    // decided we should not unroll".
    UnrollFactor = 0;
    LLVM_DEBUG(DBGS() << "got advice nounroll\n");
  }

  return std::make_unique<UnrollAdvice>(this, UnrollFactor);
}
