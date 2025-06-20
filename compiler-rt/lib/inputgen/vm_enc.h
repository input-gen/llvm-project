#include <algorithm>
#include <bit>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <sys/types.h>
#include <tuple>
#include <type_traits>

#include "common.h"
#include "logging.h"

#ifndef VM_ENC_H
#define VM_ENC_H

namespace __ig {

enum AccessKind { READ, WRITE, CHECK_INITIALIZED, BCI_READ };

enum BitsKind {
  InitBit,
  PtrBit,
  RecordBit,
  SavedBit,
};
// TODO: I am unsure if the first row second element (remainder 1), should be
// 0x0N or 0xN0.
static uint64_t BitsTable[4][5][2] = {
    {
        {0x00000001, 0x10},
        {0x00000011, 0x0},
        {0x00001111, 0x0},
        {0x0, 0x0},
        {0x11111111, 0x0},
    },
    {
        {0x00000002, 0x20},
        {0x00000022, 0x0},
        {0x00002222, 0x0},
        {0x0, 0x0},
        {0x22222222, 0x0},
    },
    {
        {0x00000004, 0x40},
        {0x00000044, 0x0},
        {0x00004444, 0x0},
        {0x0, 0x0},
        {0x44444444, 0x0},
    },
    {
        {0x00000008, 0x80},
        {0x00000088, 0x0},
        {0x00008888, 0x0},
        {0x0, 0x0},
        {0x88888888, 0x0},
    },
};

template <typename T, typename MASK_TYPE = typename std::remove_const<T>::type>
static constexpr int leadingN(const T &DATA) {
  int BITSN{sizeof(T) * CHAR_BIT}, I{BITSN};
  MASK_TYPE MASK{1u << (BITSN - 1)};
  for (; I && !(DATA & MASK); I--, MASK >>= 1) {
  }
  return BITSN - I;
}

static constexpr uint32_t NumEncodingBits = 2;
constexpr uint32_t MAGIC = 0b101;
static constexpr uint32_t NumMagicBits = (8 * sizeof(MAGIC)) - leadingN(MAGIC);

struct ObjectManager;

extern std::function<void(uint32_t)> ErrorFn;
[[noreturn]] void error(uint32_t ErrorCode);

struct EncodingSchemeTy {
  ObjectManager &OM;
  EncodingSchemeTy(ObjectManager &OM) : OM(OM) {}

  union EncTy {
    char *VPtr;
    struct __attribute__((packed)) {
      uint64_t Bits : (sizeof(char *) * 8) - NumEncodingBits;
      uint32_t EncodingId : NumEncodingBits;
    } Bits;
    EncTy(char *VPtr) : VPtr(VPtr) {}
  };

  static uint32_t getEncoding(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.EncodingId;
  }
};

template <uint32_t EncodingNo, uint32_t OffsetBits, uint32_t BucketBits,
          uint32_t RealPtrBits>
struct BucketSchemeTy : public EncodingSchemeTy {
  BucketSchemeTy(ObjectManager &OM) : EncodingSchemeTy(OM) {}
  ~BucketSchemeTy() {
    INPUTGEN_DEBUG(fprintf(stderr, "Buckets used: %i\n", NumBucketsUsed));
  }

  static constexpr uint32_t ThisEncodingNo = EncodingNo;

  static constexpr uint32_t NumOffsetBits = OffsetBits;
  static constexpr uint32_t NumBucketBits = BucketBits;
  static constexpr uint32_t NumRealPtrBits = RealPtrBits;

  static_assert(NumEncodingBits + NumMagicBits + NumOffsetBits * 2 +
                        NumBucketBits + NumRealPtrBits ==
                    (8 * sizeof(char *)),
                "Size missmatch!");

  static constexpr uint32_t NumBuckets = 1 << BucketBits;
  uint64_t Buckets[NumBuckets];
  uint32_t NumBucketsUsed = 0;

  void reset() {
    for (uint32_t I = 0; I < NumBucketsUsed; ++I)
      Buckets[I] = 0;
    NumBucketsUsed = 0;
  }

  union EncTy {
    char *VPtr;
    struct __attribute__((packed)) {
      int32_t Offset : NumOffsetBits;
      uint32_t Magic : NumMagicBits;
      uint32_t Size : NumOffsetBits;
      uint32_t BuckedIdx : NumBucketBits;
      uint32_t RealPtr : NumRealPtrBits;
      uint32_t EncodingId : NumEncodingBits;
    } Bits;
    static_assert(sizeof(Bits) == sizeof(char *), "bad size");

    EncTy(uint32_t Size, uint32_t BuckedIdx, uint32_t RealPtr) {
      Bits.Offset = 0;
      Bits.Magic = MAGIC;
      Bits.Size = Size;
      Bits.BuckedIdx = BuckedIdx;
      Bits.RealPtr = RealPtr;
      Bits.EncodingId = EncodingNo;
    }
    EncTy(char *VPtr) : VPtr(VPtr) {}
  };
  static_assert(sizeof(EncTy) == sizeof(char *), "bad size");

  static constexpr uint32_t NumBucketValueBits =
      (8 * sizeof(char *) - NumRealPtrBits);
  static_assert(NumBucketValueBits <= sizeof(Buckets[0]) * 8,
                "Bucket value too large!");

  union DecTy {
    char *Ptr;
    struct __attribute__((packed)) {
      uint32_t RealPtr : NumRealPtrBits;
      uint32_t BucketValue : NumBucketValueBits;
    } Bits;

    DecTy(char *Ptr) : Ptr(Ptr) {}
    DecTy(uint32_t RealPtr, uint32_t BucketValue) {
      Bits.RealPtr = RealPtr;
      Bits.BucketValue = BucketValue;
    }
  };
  static_assert(sizeof(DecTy) == sizeof(char *), "bad size");

  char *encode(char *Ptr, uint32_t Size) {
    assert(Size < (1ULL << NumOffsetBits));
    DecTy D(Ptr);
    uint32_t BucketIdx = ~0u;
    for (uint32_t Idx = 0; Idx < NumBucketsUsed; ++Idx) {
      if (Buckets[Idx] == D.Bits.BucketValue) {
        BucketIdx = Idx;
        break;
      }
    }
    if (BucketIdx == ~0u) {
      if (NumBucketsUsed == NumBuckets) {
        fprintf(stderr, "out of buckets!\n");
        error(1000);
        std::terminate();
      }
      BucketIdx = NumBucketsUsed++;
      Buckets[BucketIdx] = D.Bits.BucketValue;
    }
    EncTy E(Size, BucketIdx, D.Bits.RealPtr);
    return E.VPtr;
  }

  char *decode(char *VPtr) {
    EncTy E(VPtr);
    DecTy D(E.Bits.RealPtr, Buckets[E.Bits.BuckedIdx]);
    return D.Ptr + E.Bits.Offset;
  }

  char *access(char *VPtr, uint32_t AccessSize, uint32_t TypeId, bool Write) {
    EncTy E(VPtr);
    DecTy D(E.Bits.RealPtr, Buckets[E.Bits.BuckedIdx]);
    if (E.Bits.Offset < 0 || E.Bits.Offset + AccessSize > E.Bits.Size) {
      fprintf(stderr,
              "Small user object memory out-of-bound %i vs %i! (Base %p)\n",
              E.Bits.Offset, E.Bits.Size, D.Ptr);
      error(1001);
      std::terminate();
    }
    return D.Ptr + E.Bits.Offset;
  }

  bool checkSize(char *VPtr, uint32_t AccessSize) {
    return access(VPtr, AccessSize, 0, 0);
  }

  bool isMagicIntact(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.Magic == MAGIC;
  }

  std::pair<int32_t, int32_t> getPtrInfo(char *VPtr) { return {-1, -1}; }
  char *getBasePtrInfo(char *VPtr) {
    return (char *)((uint64_t)getBase(VPtr) | (uint64_t)EncodingNo);
  }
  char *getBase(char *VPtr) {
    EncTy E(VPtr);
    DecTy D(E.Bits.RealPtr, Buckets[E.Bits.BuckedIdx]);
    return D.Ptr;
  }
  char *getBaseVPtr(char *VPtr) {
    EncTy ED(VPtr);
    return VPtr - ED.Bits.Offset;
  }
};

template <uint32_t EncodingNo, uint32_t ObjectBits>
struct BigObjSchemeTy : public EncodingSchemeTy {
  BigObjSchemeTy(ObjectManager &OM) : EncodingSchemeTy(OM) {}
  ~BigObjSchemeTy() {
    INPUTGEN_DEBUG(fprintf(stderr, "Buckets used: %i\n", NumObjectsUsed));
  }

  struct ObjDescTy {
    char *Base;
    uint64_t Size;
  };

  static constexpr uint32_t NumObjectBits = ObjectBits;
  static constexpr uint32_t NumOffsetBits =
      (sizeof(void *) * 8) - NumObjectBits - NumEncodingBits - NumMagicBits;
  static constexpr uint32_t NumObjects = 1 << ObjectBits;
  ObjDescTy Objects[NumObjects];
  uint32_t NumObjectsUsed = 0;

  void reset() { NumObjectsUsed = 0; }

  union EncTy {
    char *VPtr;
    struct __attribute__((packed)) {
      int64_t Offset : NumOffsetBits;
      uint32_t Magic : NumMagicBits;
      uint32_t ObjectIdx : NumObjectBits;
      uint32_t EncodingId : NumEncodingBits;
    } Bits;
    static_assert(sizeof(Bits) == sizeof(char *), "bad size");

    EncTy(uint32_t Size, uint32_t ObjectIdx) {
      Bits.Offset = 0;
      Bits.Magic = MAGIC;
      Bits.ObjectIdx = ObjectIdx;
      Bits.EncodingId = EncodingNo;
    }
    EncTy(char *VPtr) : VPtr(VPtr) {}
  };
  static_assert(sizeof(EncTy) == sizeof(char *), "bad size");

  char *encode(char *Ptr, uint32_t Size) {
    assert(Size < (1ULL << NumOffsetBits));
    if (NumObjectsUsed == NumObjects) {
      fprintf(stderr, "out of objects!\n");
      error(1000);
      std::terminate();
    }
    uint32_t ObjectIdx = NumObjectsUsed++;
    Objects[ObjectIdx] = {Ptr, Size};
    EncTy E(Size, ObjectIdx);
    return E.VPtr;
  }

  char *decode(char *VPtr) {
    EncTy E(VPtr);
    auto [Base, Size] = Objects[E.Bits.ObjectIdx];
    return Base + E.Bits.Offset;
  }

  char *access(char *VPtr, uint32_t AccessSize, uint32_t TypeId, bool Write) {
    EncTy E(VPtr);
    auto [Base, Size] = Objects[E.Bits.ObjectIdx];
    if (E.Bits.Offset < 0 || E.Bits.Offset + AccessSize > (int64_t)Size) {
      fprintf(stderr,
              "Large user memory out-of-bound %lli vs %lli (Base %p)!\n",
              E.Bits.Offset, Size, Base);
      error(1001);
      std::terminate();
    }
    return Base + E.Bits.Offset;
  }

  bool checkSize(char *VPtr, uint32_t AccessSize) {
    return access(VPtr, AccessSize, 0, 0);
  }

  bool isMagicIntact(char *VPtr) {
    EncTy E(VPtr);
    return E.Bits.Magic == MAGIC;
  }

  std::pair<int32_t, int32_t> getPtrInfo(char *VPtr) { return {-1, -1}; }
  char *getBasePtrInfo(char *VPtr) {
    return (char *)((uint64_t)getBase(VPtr) | (uint64_t)EncodingNo);
  }
  char *getBase(char *VPtr) {
    EncTy E(VPtr);
    auto [Base, Size] = Objects[E.Bits.ObjectIdx];
    return Base;
  }
  char *getBaseVPtr(char *VPtr) {
    EncTy E(VPtr);
    return VPtr - E.Bits.Offset;
  }
};

template <uint32_t EncodingNo, uint32_t OffsetBits>
struct TableSchemeTy : public EncodingSchemeTy {
  static constexpr uint32_t NumOffsetBits = OffsetBits;
  static constexpr uint32_t NumTableIdxBits =
      (sizeof(char *) * 8) - NumOffsetBits - NumMagicBits - NumEncodingBits;
  static constexpr uint32_t DefaultOffset = 1 << (NumOffsetBits - 1);

  struct __attribute__((packed)) SeedTy {
    uint32_t Begin;
    int32_t Increment;
    SeedTy() = delete;
    SeedTy(uint32_t Begin, int32_t Increment)
        : Begin(Begin), Increment(Increment) {}
  };
  template <typename T> struct GenDistribution {
    GenDistribution() = delete;
    GenDistribution(T Min, T Max) : Min(Min), Max(Max) {}

    using OffsetTy = int32_t;
    T Min, Max;
    static_assert(std::is_signed<T>::value);
    T get(SeedTy Seed, OffsetTy Offset) {
      if constexpr (std::is_integral<T>::value)
        return ((Seed.Begin + Seed.Increment * Offset) % (Max - Min)) + Min;
      else if constexpr (std::is_floating_point<T>::value)
        // TODO this fmod can potentially be slow? But I think it probably gets
        // lost in the sea of memory accesses we make so it should be fine.
        return (fmod(Seed.Begin + Seed.Increment * Offset, Max - Min)) + Min;
      else
        static_assert(false);
    }
  };
  GenDistribution<int64_t> IntDistribution;
  GenDistribution<float> FloatDistribution;

  struct TableEntryTy {
    char *Base;
    char *Shadow;
    int32_t NegativeSize;
    bool AnyRead = false;
    bool AnyAccess = false;
    bool IsNull = false;
    bool AnyPtrRead = false;
    char *SavedValues = nullptr;
    char *GlobalName = nullptr;
    SeedTy Seed;

    static uint64_t getTotalSizeStatic(uint64_t Size) {
      return Size + getShadowSizeStatic(Size);
    }
    static uint64_t getShadowSizeStatic(uint64_t Size) {
      return (Size + 1) / 2;
    }

    static void checkSizes(int32_t PositiveSize, int32_t NegativeSize) {
      //   16        0                   32
      //   ------------------------------
      //   ^         ^                   ^
      //   |         |                   | PositiveSize = 32
      //   |         | BasePtr
      //   |
      //   | NegativeSize = 16
      //
      // Check that the lower end of the object (NegativeSize) is actually on
      // the left side of the higher end (PositiveSize)
      INPUTGEN_DEBUG(std::cerr << "check " << -NegativeSize
                               << " <= " << PositiveSize << "\n");
      assert(PositiveSize >= -NegativeSize);
    }

    TableEntryTy(char *Base, int32_t PositiveSize, int32_t NegativeSize,
                 char *GlobalName, SeedTy Seed)
        : Base(Base), Shadow(Base + PositiveSize + NegativeSize),
          NegativeSize(NegativeSize), GlobalName(GlobalName), Seed(Seed) {
      checkSizes(PositiveSize, NegativeSize);
      INPUTGEN_DEBUG(std::cerr << "TableEntryTy(" << (void *)Base << ", "
                               << PositiveSize << ", " << NegativeSize << ", "
                               << (bool)GlobalName << ")");
    }
    char *getBase() const { return Base; }
    char *getShadow() const { return Shadow; }
    uint32_t getShadowSize() const { return getShadowSizeStatic(getSize()); }
    uint32_t getSize() const { return Shadow - Base; }
    int32_t getPositiveSize() const { return Shadow - Base - NegativeSize; }
    int32_t getNegativeSize() const { return NegativeSize; }

    uint64_t getValue(TableSchemeTy<EncodingNo, OffsetBits> &TS,
                      uint32_t TypeId, uint32_t TypeSize, uint32_t Offset) {
      switch (TypeId) {
      case 2:
        return std::bit_cast<uint32_t>(TS.FloatDistribution.get(Seed, Offset));
      case 3:
        return std::bit_cast<uint64_t>(
            (double)TS.FloatDistribution.get(Seed, Offset));
      case 12:
        return TS.IntDistribution.get(Seed, Offset);
      case 14:
        return (uint64_t)TS.create(8);
      default:
        fprintf(stderr, "unknown type id %i\n", TypeId);
        error(1002);
        std::terminate();
      }
    }

    void grow(int32_t NewPositiveSize, int32_t NewNegativeSize) {
      if (GlobalName) {
        fprintf(stderr, "Out of bound access on global detected: %p; UB!\n",
                Base);
        error(1003);
        std::terminate();
      }

      // We need the sizes to always be divisible by two when resizing because
      // otherwise aligning the shadown correctly is very annoying (and
      // inefficient)
      assert(NewPositiveSize % 2 == 0);
      assert(NewNegativeSize % 2 == 0);

      checkSizes(NewPositiveSize, NewNegativeSize);
      uint32_t OldSize = getSize();
      uint32_t NewSize = NewPositiveSize + NewNegativeSize;
      int32_t NegativeDifference = NewNegativeSize - getNegativeSize();
      uint32_t NewTotalSize = getTotalSizeStatic(NewSize);
      char *NewBase;
      char *NewSavedValues;
      if (false && NegativeDifference == 0) {
        // TODO
        NewBase = (char *)realloc(Base, NewTotalSize);
        __builtin_memcpy(NewBase + NewPositiveSize + NewNegativeSize,
                         NewBase + OldSize, getShadowSize());
        __builtin_memset(NewBase + NewPositiveSize + NewNegativeSize +
                             getShadowSize(),
                         0, (NewTotalSize - NewSize) - getShadowSize());
      } else {
        NewBase = (char *)calloc(NewTotalSize, 1);
        __builtin_memcpy(NewBase + NegativeDifference, Base, OldSize);
        __builtin_memcpy(NewBase + NewSize + NegativeDifference / 2,
                         getShadow(), getShadowSize());
        free(Base);
        if (SavedValues) {
          NewSavedValues = (char *)calloc(NewSize, 1);
          __builtin_memcpy(NewSavedValues + NegativeDifference, SavedValues,
                           OldSize);
          free(SavedValues);
        } else {
          NewSavedValues = nullptr;
        }
      }
      Base = NewBase;
      Shadow = NewBase + NewPositiveSize + NewNegativeSize;
      NegativeSize = NewNegativeSize;
      INPUTGEN_DEBUG(std::cerr << "update TableEntryTy(" << (void *)getBase()
                               << ", " << getPositiveSize() << ", "
                               << getNegativeSize() << ", " << (bool)GlobalName
                               << ")\n");
    }

    void printShadow() const { dumpMemoryBinary(getShadow(), getShadowSize()); }
    void printMemory() const { dumpMemoryHex(getBase(), getSize()); }
    void printSavedValues() const {
      if (SavedValues)
        dumpMemoryHex(SavedValues, getSize());
      else
        fputs("(nil)\n", stderr);
    }
    void printStats() const {
      fprintf(stderr, "- %p:%u [%p]\n", (void *)getBase(), getSize(),
              SavedValues);
      fputs("Shadow ", stderr);
      printShadow();
      fputs("Memory ", stderr);
      printMemory();
      fputs("Saved Values ", stderr);
      printSavedValues();
    }
  };

  TableEntryTy *Table;
  uint32_t TableEntryCnt = 0;

  TableSchemeTy(ObjectManager &OM)
      : EncodingSchemeTy(OM), IntDistribution(-100, 128),
        FloatDistribution(-3.14, 4),
        Table((TableEntryTy *)malloc(sizeof(TableEntryTy) *
                                     (1 << NumTableIdxBits))) {}

  void reset() {
    // TODO reuse memory?
    for (uint32_t I = 0; I < TableEntryCnt; ++I) {
      free(Table[I].getBase());
      free(Table[I].SavedValues);
    }
    TableEntryCnt = 0;
  }

  union EncDecTy {
    char *VPtr;
    struct __attribute__((packed)) {
      uint32_t Offset : NumOffsetBits;
      uint32_t Magic : NumMagicBits;
      uint32_t TableIdx : NumTableIdxBits;
      uint32_t EncodingId : NumEncodingBits;
    } Bits;

    EncDecTy(uint32_t Offset, uint32_t TableIdx) {
      Bits.Offset = Offset;
      Bits.Magic = MAGIC;
      Bits.TableIdx = TableIdx;
      Bits.EncodingId = EncodingNo;
    }
    EncDecTy(char *VPtr) : VPtr(VPtr) {}
  };

  static_assert(sizeof(EncDecTy) == sizeof(char *), "bad size");

  SeedTy getRTObjSeed();

  char *create(uint32_t Size, char *GlobalName = nullptr) {
    auto TEC = TableEntryCnt++;
    int32_t NegativeSize = 0;
    int32_t PositiveSize = GlobalName ? Size : Size * 8;
    uint32_t TotalSize = TableEntryTy::getTotalSizeStatic(PositiveSize);
    char *Base = (char *)calloc(TotalSize, 1);
    Table[TEC] = TableEntryTy(Base, PositiveSize, NegativeSize, GlobalName,
                              getRTObjSeed());
    EncDecTy ED(DefaultOffset, TEC);
    INPUTGEN_DEBUG(std::cerr << " --> " << (void *)ED.VPtr << "\n");
    return ED.VPtr;
  }

  char *decode(char *VPtr) {
    EncDecTy ED(VPtr);
    TableEntryTy &TE = Table[ED.Bits.TableIdx];
    if (TE.IsNull)
      return nullptr;
    int32_t RelOffset = (uint32_t)ED.Bits.Offset - DefaultOffset;
    return TE.Base + RelOffset;
  }

  __attribute__((always_inline)) uint64_t
  readVariableSize(char *MPtr, uint32_t AccessSize) {
    switch (AccessSize) {
    case 0:
    case 1:
      return *MPtr;
    case 2:
      return *(uint16_t *)MPtr;
    case 4:
      return *(uint32_t *)MPtr;
    case 8:
      return *(uint64_t *)MPtr;
    default:
      std::cerr << "unexpected access size " << AccessSize << "\n";
      __builtin_trap();
    }
  }

  __attribute__((always_inline)) void
  writeVariableSize(char *MPtr, uint32_t AccessSize, uint64_t Value) {
    switch (AccessSize) {
    case 0:
    case 1:
      *MPtr = Value;
      break;
    case 2:
      *(uint16_t *)MPtr = Value;
      break;
    case 4:
      *(uint32_t *)MPtr = Value;
      break;
    case 8:
      *(uint64_t *)MPtr = Value;
      break;
    default:
      std::cerr << "unexpected access size " << AccessSize << "\n";
      __builtin_trap();
    }
  }

  __attribute__((always_inline)) void
  checkAndWrite(TableEntryTy &TE, char *MemP, char *ShadowP,
                uint32_t AccessSize, uint32_t TypeId, AccessKind AK,
                uint32_t Rem, bool &AnyInitialized, bool &AllInitialized,
                int32_t Offset) {
    assert(AccessSize <= 8 && std::has_single_bit(AccessSize));

    if (TE.IsNull) {
      AnyInitialized = true;
      if (AK == CHECK_INITIALIZED || AK == BCI_READ)
        return;
      fprintf(stderr, "access to nullptr (object) detected: %p; UB!\n", MemP);
      error(1002);
      std::terminate();
    }

    uint64_t ShadowVal = readVariableSize(ShadowP, AccessSize / 2);
    uint64_t InitBits = BitsTable[InitBit][AccessSize / 2][Rem];
    uint64_t ShadowInit = ShadowVal & InitBits;
    bool FullInit = ShadowInit == InitBits;
    bool PartialInit = ShadowInit;
    AllInitialized &= FullInit;
    AnyInitialized |= PartialInit;
    if (AK == CHECK_INITIALIZED)
      return;

    TE.AnyAccess = true;
    uint64_t PtrBits = BitsTable[PtrBit][AccessSize / 2][Rem];
    uint64_t RecordBits = BitsTable[RecordBit][AccessSize / 2][Rem];
    uint64_t SavedBits = BitsTable[SavedBit][AccessSize / 2][Rem];

    uint64_t FullRecord = (ShadowVal & RecordBits) == RecordBits;
    uint64_t NoneSaved = !(ShadowVal & SavedBits);
    [[maybe_unused]] uint64_t PartialSaved =
        ((ShadowVal & SavedBits) != SavedBits) && !NoneSaved;
    [[maybe_unused]] uint64_t FullSaved = (ShadowVal & SavedBits) == SavedBits;
    uint64_t PartiallyUnsaved =
        ((ShadowVal & RecordBits) << 1) != (ShadowVal & SavedBits);

    ShadowVal |= InitBits;

    if (!FullInit) {
      if (AK == READ) {
        TE.AnyRead = true;
        if (TypeId == 14)
          TE.AnyPtrRead = true;
      }
    }

    // Order:
    // - not initialized first,
    // - [skipped] a fully initialized read,
    // - a partially initialized read,
    // - [skipped] a write over a non-recorded region,
    // - a write over an entire recorded region without any saved parts,
    // - a write over partially recorded unsaved parts.
    if (!PartialInit) {
      if (AK != WRITE) {
        ShadowVal |= RecordBits | ((TypeId == 14) ? PtrBits : 0);
        if (AK != BCI_READ)
          writeVariableSize(MemP, AccessSize,
                            TE.getValue(*this, TypeId, AccessSize, Offset));
      }
    } else if (AK != WRITE && !FullInit) {
      assert(!FullInit && PartialInit);
      ShadowVal |= RecordBits | ((TypeId == 14) ? PtrBits : 0);
      // TODO: PtrBits?
      if (AK != BCI_READ) {
        __builtin_assume(AccessSize <= 8);
        uint64_t SingleInitBits = BitsTable[InitBits][0][0];
        for (uint32_t Byte = 0; Byte < AccessSize; ++Byte) {
          uint64_t InitByteBits = SingleInitBits << (AccessSize - Byte - 1);
          if (!(ShadowVal & InitByteBits))
            writeVariableSize(MemP + Byte, 1,
                              TE.getValue(*this, TypeId, 1, Offset));
        }
      }
    } else if (AK == WRITE && FullRecord && NoneSaved) {
      if (!TE.SavedValues)
        TE.SavedValues = (char *)calloc(TE.getSize(), 1);
      __builtin_memcpy(TE.SavedValues + (MemP - TE.getBase()), MemP,
                       AccessSize);
      ShadowVal |= SavedBits;
    } else if (AK == WRITE && PartiallyUnsaved) {
      assert(PartialInit && !FullSaved && (PartialSaved || !FullRecord));
      if (!TE.SavedValues)
        TE.SavedValues = (char *)calloc(TE.getSize(), 1);
      __builtin_assume(AccessSize <= 8);
      uint64_t SingleSavedBits = BitsTable[SavedBit][0][0];
      uint64_t SingleRecordBits = BitsTable[RecordBit][0][0];
      for (uint32_t Byte = 0; Byte < AccessSize; ++Byte) {
        uint64_t SavedByteBits = SingleSavedBits << (4 * Byte);
        uint64_t RecordByteBits = SingleRecordBits << (4 * Byte);
        if (!(ShadowVal & RecordByteBits) || (ShadowVal & SavedByteBits))
          continue;
        __builtin_memcpy(TE.SavedValues + (MemP - TE.getBase() + Byte),
                         MemP + Byte, 1);
        ShadowVal |= SavedByteBits;
      }
    }

    writeVariableSize(ShadowP, AccessSize / 2, ShadowVal);
    assert((ShadowVal & InitBits) == InitBits);
    if (!((readVariableSize(ShadowP, AccessSize / 2) & InitBits) == InitBits)) {
      printf("%llu %llu %llu\n", InitBits,
             readVariableSize(ShadowP, AccessSize / 2),
             (readVariableSize(ShadowP, AccessSize / 2) & InitBits));
    }
    assert((readVariableSize(ShadowP, AccessSize / 2) & InitBits) == InitBits);
  }
  __attribute__((always_inline)) char *access(char *VPtr, uint32_t AccessSize,
                                              uint32_t TypeId, AccessKind AK,
                                              bool &AnyInitialized,
                                              bool &AllInitialized) {
    EncDecTy ED(VPtr);
    TableEntryTy &TE = Table[ED.Bits.TableIdx];

    int32_t RelOffset = (uint32_t)ED.Bits.Offset - DefaultOffset;
    INPUTGEN_DEBUG(std::cerr << "access at offset " << RelOffset << "\n");

    auto Size = TE.getSize();
    int32_t PositiveSize = TE.getPositiveSize();
    int32_t NegativeSize = TE.getNegativeSize();
    if (-RelOffset > NegativeSize) [[unlikely]] {
      uint32_t Overshot = -RelOffset - NegativeSize;
      uint32_t NewNegativeSize =
          std::max(4 * Overshot, 4 * Size) + NegativeSize;
      // assert(std::has_single_bit(NewNegativeSize));
      TE.grow(PositiveSize, NewNegativeSize);
      PositiveSize = TE.getPositiveSize();
      NegativeSize = TE.getNegativeSize();
    } else if (RelOffset + (int32_t)AccessSize > PositiveSize) [[unlikely]] {
      uint32_t Overshot = (RelOffset + AccessSize) - PositiveSize;
      uint32_t NewPositiveSize =
          std::max(4 * Overshot, 4 * Size) + PositiveSize;
      // TODO do we need this?
      // assert(std::has_single_bit(NewPositiveSize));
      TE.grow(NewPositiveSize, NegativeSize);
      PositiveSize = TE.getPositiveSize();
      NegativeSize = TE.getNegativeSize();
    }
    auto OffsetFromBase = RelOffset + NegativeSize;
    INPUTGEN_DEBUG(std::cerr << "offset from base " << OffsetFromBase << "\n");
    auto Div = OffsetFromBase >> 1;
    auto Mod = OffsetFromBase & 1;
    char *ShadowP = (TE.getShadow() + Div);
    char *MemP = (TE.Base + OffsetFromBase);

    if (Mod) [[unlikely]] {
      checkAndWrite(TE, MemP, ShadowP, 1, TypeId, AK, Mod, AnyInitialized,
                    AllInitialized, RelOffset);
      MemP += 1;
      ShadowP += 1;
      AccessSize -= 1;
    }
    if (AccessSize == 8) [[likely]]
      checkAndWrite(TE, MemP, ShadowP, 8, TypeId, AK, 0, AnyInitialized,
                    AllInitialized, RelOffset);
    else if (AccessSize == 4) [[likely]]
      checkAndWrite(TE, MemP, ShadowP, 4, TypeId, AK, 0, AnyInitialized,
                    AllInitialized, RelOffset);
    else [[unlikely]] {
      for (uint32_t Bytes : {8, 4, 2}) {
        while (AccessSize >= Bytes) {
          checkAndWrite(TE, MemP, ShadowP, Bytes, TypeId, AK, 0, AnyInitialized,
                        AllInitialized, RelOffset);
          MemP += Bytes;
          ShadowP += (Bytes / 2);
          AccessSize -= Bytes;
        }
      }
      if (AccessSize)
        checkAndWrite(TE, MemP, ShadowP, AccessSize, TypeId, AK, 0,
                      AnyInitialized, AllInitialized, RelOffset);
    }

    return (TE.Base + OffsetFromBase);
  }

  bool isMagicIntact(char *VPtr) {
    EncDecTy ED(VPtr);
    return ED.Bits.Magic == MAGIC;
  }

  std::pair<int32_t, int32_t> getPtrInfo(char *VPtr) {
    EncDecTy ED(VPtr);
    int32_t RelOffset = (uint32_t)ED.Bits.Offset - DefaultOffset;
    return {(uint32_t)ED.Bits.TableIdx, RelOffset};
  }
  char *getBasePtrInfo(char *VPtr) {
    return (char *)((uint64_t)getBase(VPtr) | (uint64_t)EncodingNo);
  }
  char *getBase(char *VPtr) {
    EncDecTy ED(VPtr);
    TableEntryTy &TE = Table[ED.Bits.TableIdx];
    return TE.Base;
  }
  char *getBaseVPtr(char *VPtr) {
    EncDecTy ED(VPtr);
    return VPtr - ED.Bits.Offset;
  }
};

} // namespace __ig

#endif
