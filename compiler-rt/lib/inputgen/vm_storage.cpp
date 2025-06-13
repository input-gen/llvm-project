
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <type_traits>

#include "vm_storage.h"
#include "common.h"
#include "global_manager.h"
#include "vm_obj.h"
#include "storage_utils.h"

using namespace __ig;

namespace __ig::storage {

struct GlobalComp {
  bool operator()(const GlobalManager::GlobalTy &G, const std::string &Name) {
    return G.Name < Name;
  }
  bool operator()(const std::string &Name, const GlobalManager::GlobalTy &G) {
    return Name < G.Name;
  }
};

Global::Global(std::ifstream &IFS, GlobalManager &GM) {
  DEFINE_READV(IFS);
  uint32_t NameSize;
  READV(NameSize);
  Name.resize(NameSize);
  READMEM(IFS, Name.data(), NameSize);
  auto Found = std::lower_bound(GM.Globals.begin(), GM.Globals.end(), Name,
                                GlobalComp{});
  if (Found == GM.Globals.end() || Found->Name != Name) {
    R.init(Range(IFS, nullptr, Name));
  } else {
    R.init(Range(IFS, Found->Address, Name));
  }
}

void Global::write(std::ofstream &OFS) {
  DEFINE_WRITEV(OFS);
  uint32_t NameSize = Name.size();
  WRITEV(NameSize);
  OFS.write(Name.data(), NameSize);
  R->write(OFS);
}

Range::Range(std::ifstream &IFS, char *Memory, const std::string &Name) {
  DEFINE_READV(IFS);
  READV(ObjIdx);
  READV(NegativeSize);
  READV(AnyRecorded);
  ptrdiff_t Length;
  READV(Length);
  // TODO I am unsure if this offset it right
  Begin = Memory + NegativeSize;
  End = Begin + Length;
  if (Length) {
    if (AnyRecorded) {
      if (Memory) {
        std::cerr << "Could not find recorded global with name " << Name
                  << "\n";
        abort();
      }
      READMEM(IFS, Begin, Length);
    }
  } else {
    Begin = End = nullptr;
  }
}

Range::Range(std::ifstream &IFS) {
  DEFINE_READV(IFS);
  READV(ObjIdx);
  READV(NegativeSize);
  READV(AnyRecorded);
  ptrdiff_t Length;
  READV(Length);
  if (Length) {
    Begin = (char *)malloc(Length);
    INPUTGEN_DEBUG(std::cerr << "malloc -- " << (void *)Begin << "\n");
    End = Begin + Length;
    if (AnyRecorded) {
      READMEM(IFS, Begin, Length);
      INPUTGEN_DEBUG({
        if (getenv("PRINT_RUNTIME_OBJECTS"))
          dumpMemoryHex(Begin, Length);
      });
    }
  } else {
    Begin = End = nullptr;
  }
}

void Range::write(std::ofstream &OFS) {
  DEFINE_WRITEV(OFS);
  WRITEV(ObjIdx);
  WRITEV(NegativeSize);
  WRITEV(AnyRecorded);
  ptrdiff_t Length = End - Begin;
  WRITEV(Length);
  if (Length)
    if (AnyRecorded)
      OFS.write(Begin, Length);
}

Ptr::Ptr(std::ifstream &IFS) {
  DEFINE_READV(IFS);
  READV(ObjIdx);
  READV(Offset);
  READV(TgtObjIdx);
  READV(TgtOffset);
}

void Ptr::write(std::ofstream &OFS) {
  DEFINE_WRITEV(OFS);
  WRITEV(ObjIdx);
  WRITEV(Offset);
  WRITEV(TgtObjIdx);
  WRITEV(TgtOffset);
}

StorageManager::StorageManager() { std::ios::sync_with_stdio(false); }

Range StorageManager::encodeRange(ObjectManager &OM, uint32_t ObjIdx,
                                  RTObjScheme::TableEntryTy &TE) {
  if (TE.IsNull) {
    return Range(ObjIdx, false, 0, nullptr, nullptr);
  }
  char *ValueP = TE.getBase();
  char *SavedP = TE.SavedValues;
  // If we have started to save values, ensure all are saved.
  char *ShadowP = TE.getShadow();
  uint64_t NoLastPtr = ~0 - sizeof(void *);
  uint64_t LastPtr = NoLastPtr;
  auto ShadowSize = TE.getShadowSize();
  bool AnyRead = TE.AnyRead;
  bool AnyPtrRead = TE.AnyPtrRead;
  for (uint32_t I = 0; (AnyRead || AnyPtrRead) && I < ShadowSize; ++I) {
    unsigned char V = ShadowP[I];
    for (auto Offset : {0, 1}) {
      uint32_t ValueI = 2 * I + Offset;
      if (AnyRead && SavedP && (V & BitsTable[RecordBit][0][Offset]) &&
          !(V & BitsTable[SavedBit][0][Offset]))
        SavedP[ValueI] = ValueP[ValueI];

      bool IsPtr = (V & BitsTable[PtrBit][0][Offset]);
      if (LastPtr + sizeof(void *) == ValueI) {
        char **PtrAddr =
            (char **)(SavedP ? SavedP + LastPtr : ValueP + LastPtr);
        auto [TgtObjIdx, TgtOffset] = OM.getPtrInfo(*PtrAddr, false);
        Ptrs.emplace_back(ObjIdx, LastPtr, TgtObjIdx, TgtOffset);
        LastPtr = NoLastPtr;
      }

      if (IsPtr && LastPtr == NoLastPtr)
        LastPtr = ValueI;
    }
  }
  if (LastPtr + sizeof(void *) == ShadowSize * 2) {
    char **PtrAddr = (char **)(SavedP ? SavedP + LastPtr : ValueP + LastPtr);
    auto [TgtObjIdx, TgtOffset] = OM.getPtrInfo(*PtrAddr, false);
    Ptrs.emplace_back(ObjIdx, LastPtr, TgtObjIdx, TgtOffset);
  }

  char *BaseP = SavedP ? SavedP : ValueP;
  return Range(ObjIdx, AnyRead, TE.getNegativeSize(), BaseP,
               BaseP + TE.getSize());
}

void StorageManager::encode(ObjectManager &OM, uint32_t ObjIdx,
                            RTObjScheme::TableEntryTy &TE) {
  Range R = encodeRange(OM, ObjIdx, TE);
  if (TE.GlobalName) {
    Globals.emplace_back(R, TE.GlobalName);
  } else {
    Ranges.push_back(R);
  }
}

void StorageManager::write(std::ofstream &OFS) {
  DEFINE_WRITEV(OFS);

  WRITEV(GenFileMagic);

  uint32_t NRanges = Ranges.size();
  WRITEV(NRanges);
  for (auto &Range : Ranges)
    Range.write(OFS);

  uint32_t NGlobals = Globals.size();
  WRITEV(NGlobals);
  for (auto &Global : Globals)
    Global.write(OFS);

  uint32_t NPtrs = Ptrs.size();
  WRITEV(NPtrs);
  for (auto &Ptr : Ptrs)
    Ptr.write(OFS);
}

void StorageManager::read(std::ifstream &IFS, GlobalManager &GM) {
  DEFINE_READV(IFS);

  GenFileMagicTy Magic;
  READV(Magic);
  if (Magic != GenFileMagic)
    FATAL();

  uint32_t NRanges;
  READV(NRanges);
  for (uint32_t I = 0; I < NRanges; ++I)
    Ranges.emplace_back(IFS);

  uint32_t NGlobals;
  READV(NGlobals);
  for (uint32_t I = 0; I < NGlobals; ++I)
    Globals.emplace_back(IFS, GM);

  uint32_t NPtrs;
  READV(NPtrs);
  for (uint32_t I = 0; I < NPtrs; ++I)
    Ptrs.emplace_back(IFS);

  for (auto &Ptr : Ptrs) {
    auto &ObjRange = Ranges[Ptr.ObjIdx];
    auto &TgtObjRange = Ranges[Ptr.TgtObjIdx];
    char *ObjPtr = &ObjRange.Begin[Ptr.Offset];
    char *TgtObjPtr =
        &TgtObjRange.Begin[Ptr.TgtOffset + TgtObjRange.NegativeSize];
    *(char **)(ObjPtr) = TgtObjPtr;
    INPUTGEN_DEBUG(std::cerr << "repoint #" << Ptr.ObjIdx << " at "
                             << (void *)ObjPtr << " --> #" << Ptr.TgtObjIdx
                             << " at " << (void *)TgtObjPtr << "\n");
  }
}

void *StorageManager::getEntryPtr() {
  return Ranges[0].Begin + Ranges[0].NegativeSize;
}

} // namespace __ig::storage
