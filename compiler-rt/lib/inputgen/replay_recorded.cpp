#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "common.h"
#include "logging.h"
#include "timer.h"
#include "storage_utils.h"

using namespace __ig;

// This means that if something was already mapped there it will not be replaced
constexpr static int fixed_flag = MAP_FIXED_NOREPLACE;
// static int fixed_flag = MAP_FIXED;

namespace __ig::storage {

struct RecordInput {
  std::string EntryName;
  void *Args;
};

std::optional<RecordInput> restore_memory(const std::string &in_file) {
  std::ifstream IFS(in_file, std::ios::binary);
  if (!IFS)
    return std::nullopt;

  DEFINE_READV(IFS);

  RecordFileMagicTy Magic;
  Magic = readV<RecordFileMagicTy>(IFS);
  if (Magic != RecordFileMagic)
    FATAL();

  uint32_t NameSize;
  READV(NameSize);
  std::string Name;
  Name.resize(NameSize);
  READMEM(IFS, Name.data(), NameSize);

  void *Args;
  IFS.read(reinterpret_cast<char *>(&Args), sizeof(Args));

  while (IFS.peek() != EOF) {
    uintptr_t start;
    size_t size;

    IFS.read(reinterpret_cast<char *>(&start), sizeof(start));
    IFS.read(reinterpret_cast<char *>(&size), sizeof(size));

    DEBUGF("READ IN START 0x%lx SIZE 0x%lx\n", start, size);

    std::vector<char> buffer(size);
    IFS.read(buffer.data(), size);

    void *addr = mmap(reinterpret_cast<void *>(start), size,
                      PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS | fixed_flag, -1, 0);

    if (addr == MAP_FAILED) {
      perror("mmap");
      return std::nullopt;
    }
    if (addr != reinterpret_cast<void *>(start)) {
      printf("mmap address does not match %p != %p\n", addr,
             reinterpret_cast<void *>(start));
      return std::nullopt;
    }

    std::memcpy(addr, buffer.data(), size);
  }

  return RecordInput{Name, Args};
}
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <file.inp> [<entry_no>]\n";
    printNumAvailableFunctions();
    printAvailableFunctions();
    exit(static_cast<int>(ExitStatus::WrongUsage));
  }

  uint32_t EntryNo = 0;
  if (argc > 2)
    EntryNo = std::atoi(argv[2]);
  if (EntryNo >= __ig_num_entry_points) {
    fprintf(stderr, "Entry %u is out of bounds, %u available\n", EntryNo,
            __ig_num_entry_points);
    exit(static_cast<int>(ExitStatus::EntryNoOutOfBounds));
  }

  std::string Input = argv[1];

  __ig::storage::RecordInput RI;
  {
    Timer T("init");
    auto res = __ig::storage::restore_memory(Input);
    if (!res) {
      std::cerr << "Memory restore failed.\n";
      return 1;
    }
    RI = *res;
  }
  std::cerr << "Memory successfully restored from " << Input << "\n";
  INPUTGEN_DEBUG(std::cerr << "Args: " << RI.Args << "\n");
  assert(RI.Args);

  if (RI.EntryName != __ig_entry_point_names[EntryNo]) {
    fprintf(stderr, "Entry %u's name %s does not match input file's entry name %s\n", EntryNo, RI.EntryName.c_str(), __ig_entry_point_names[EntryNo]);
    exit(static_cast<int>(ExitStatus::EntryNoOutOfBounds));
  }

  std::cerr << "Replaying " << RI.EntryName << " with input " << Input << "\n";

  {
    Timer T("replay");
    __ig_entry(EntryNo, RI.Args);
  }

  return 0;
}
