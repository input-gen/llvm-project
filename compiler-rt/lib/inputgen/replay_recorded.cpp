#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

constexpr static int fixed_flag = MAP_FIXED_NOREPLACE;
// static int fixed_flag = MAP_FIXED;

std::optional<void *> restore_memory(const std::string &in_file) {
  std::ifstream in(in_file, std::ios::binary);
  if (!in)
    return {};

  void *Args;
  in.read(reinterpret_cast<char *>(&Args), sizeof(Args));

  while (in.peek() != EOF) {
    uintptr_t start;
    size_t size;

    in.read(reinterpret_cast<char *>(&start), sizeof(start));
    in.read(reinterpret_cast<char *>(&size), sizeof(size));

    std::vector<char> buffer(size);
    in.read(buffer.data(), size);

    void *addr = mmap(reinterpret_cast<void *>(start), size,
                      PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS | fixed_flag, -1, 0);

    if (addr == MAP_FAILED) {
      perror("mmap");
      return {};
    }

    std::memcpy(addr, buffer.data(), size);
  }

  return Args;
}

int main(int argc, char *argv[]) {
  std::string input = "/home/ivan/tmp/inputgen_record_outfile";

  auto res = restore_memory(input);
  if (!res) {
    std::cerr << "Memory restore failed.\n";
    return 1;
  }

  std::cout << "Memory successfully restored from " << input << "\n";
  std::cout << "Args: " << *res << "\n";
  std::cout << "Press Enter to continue...\n";
  std::cin.get();

  return 0;
}
