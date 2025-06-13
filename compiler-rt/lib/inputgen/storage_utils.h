#ifndef STORAGE_UTILS_H_
#define STORAGE_UTILS_H_

#include <cstdint>

#include "logging.h"

namespace __ig::storage {

// INPUTGEN
using GenFileMagicTy = uint64_t;
static constexpr GenFileMagicTy GenFileMagic = 0x4e45475455504e49;

using RecordFileMagicTy = uint64_t;
static constexpr RecordFileMagicTy RecordFileMagic = 0x0e45475455504e49;

static void FATAL(std::string Reason = "") {
  std::cerr << "Malformed input!\n";
  if (!Reason.empty())
    std::cerr << Reason << "\n";
  exit(1);
}

template <typename T> static const char *ccast(const T *Ptr) {
  return reinterpret_cast<const char *>(Ptr);
}

template <typename T> static char *ccast(T *Ptr) {
  return reinterpret_cast<char *>(Ptr);
}

template <typename T> static T readV(std::ifstream &Input) {
  if (Input.eof())
    FATAL("Expected more input.");
  T El;
  Input.read(ccast(&El), sizeof(El));
  if (Input.gcount() != sizeof(El))
    FATAL("Expected more input.");
  return El;
}

[[maybe_unused]]
static void READMEM(std::ifstream &Input, char *Mem, size_t Size) {
  if (Input.eof())
    FATAL("Expected more input.");
  Input.read(Mem, Size);
  if ((size_t)Input.gcount() != Size)
    FATAL("Expected more input.");
}

template <typename T> static T writeV(std::ofstream &Output, T El) {
  Output.write(ccast(&El), sizeof(El));
  return El;
}

#ifndef NDEBUG
#define DEFINE_READV(S)                                                        \
  do {                                                                         \
  } while (0)
#define DEFINE_WRITEV(S)                                                       \
  do {                                                                         \
  } while (0)
#define READV(V)                                                               \
  do {                                                                         \
    V = readV<decltype(V)>(IFS);                                               \
    INPUTGEN_DEBUG(std::cerr << "read " #V " " << V << "\n");                  \
  } while (0)
#define WRITEV(V)                                                              \
  do {                                                                         \
    writeV<decltype(V)>(OFS, V);                                               \
    INPUTGEN_DEBUG(std::cerr << "wrote " #V " " << V << "\n");                 \
  } while (0)
#else
#define DEFINE_READV(S) auto READV = [&S]<typename T>(T &V) { V = readV<T>(S); }
#define DEFINE_WRITEV(S) auto WRITEV = [&S]<typename T>(T V) { writeV(S, V); };
#endif

} // namespace __ig::storage

#endif // STORAGE_UTILS_H_
