//===--- record.cpp - Emit LLVM Code from ASTs for a Module ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the runtime for recording inputs.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "common.h"
#include "storage_utils.h"

using namespace __ig;

namespace __ig::storage {

struct MemRegion {
  uintptr_t start;
  uintptr_t end;
  std::string perms;
};

std::vector<MemRegion> parse_maps(pid_t pid) {
  std::vector<MemRegion> regions;
  std::ifstream maps("/proc/" + std::to_string(pid) + "/maps");
  std::string line;

  while (std::getline(maps, line)) {
    std::istringstream iss(line);
    std::string addr_range, perms;
    if (!(iss >> addr_range >> perms))
      continue;

    auto dash = addr_range.find('-');
    uintptr_t start = std::stoul(addr_range.substr(0, dash), nullptr, 16);
    uintptr_t end = std::stoul(addr_range.substr(dash + 1), nullptr, 16);

    if (perms[0] == 'r') {
      regions.push_back({start, end, perms});
    }
  }

  return regions;
}

size_t read_all(std::ifstream &mem_stream, uintptr_t addr, char *buffer,
                size_t size) {
  mem_stream.clear(); // Clear any error flags (EOF, failbit, etc.)
  mem_stream.seekg(addr, std::ios::beg);
  if (!mem_stream.good())
    return false;

  size_t total = 0;
  while (total < size) {
    mem_stream.read(buffer + total, size - total);
    std::streamsize bytes_read = mem_stream.gcount();
    if (bytes_read <= 0)
      break;
    total += bytes_read;
  }

  return total;
}

bool dump_memory(pid_t pid, const std::string &out_file, const char *NameC,
                 void *Args) {
  auto regions = parse_maps(pid);
  uintptr_t ArgsUint = reinterpret_cast<uintptr_t>(Args);
  std::ofstream OFS(out_file, std::ios::binary);
  if (!OFS)
    return false;

  DEFINE_WRITEV(OFS);


  WRITEV(RecordFileMagic);

  std::string Name = NameC;
  uint32_t NameSize = Name.size();
  WRITEV(NameSize);
  OFS.write(Name.data(), NameSize);

  OFS.write(reinterpret_cast<const char *>(&Args), sizeof(Args));

  std::ifstream mem("/proc/" + std::to_string(pid) + "/mem", std::ios::binary);
  if (!mem) {
    std::cerr << "Failed to open mem file\n";
    return false;
  }

  bool FoundArgs = false;
  for (const auto &region : regions) {
    size_t size = region.end - region.start;
    assert(size > 0);
    std::vector<char> buffer(size);

    size_t read_bytes = read_all(mem, region.start, buffer.data(), size);
    if (read_bytes != size) {
      std::cerr << "Warning: Partial or failed read at region 0x" << std::hex
                << region.start << "-0x" << region.end << "\n";
    }
    DEBUGF("WRITE IN START 0x%lx SIZE 0x%lx\n", region.start,
           region.end - region.start);
    OFS.write(reinterpret_cast<const char *>(&region.start),
              sizeof(region.start));
    OFS.write(reinterpret_cast<const char *>(&size), sizeof(size));
    OFS.write(buffer.data(), size);
    if (ArgsUint >= region.start && ArgsUint < region.end) {
      FoundArgs = true;
      DEBUGF("Args found\n");
    }
  }

  if (!FoundArgs) {
    puts("Did not find args\n");
    return false;
  }

  return true;
}
} // namespace __ig::storage

IG_API_ATTRS
void __ig_record_push(const char *Name, void *Args) {
  printf("Starting recording of %s\n", Name);
  pid_t parent_pid = getpid();
  const char *outfile = "/home/ivan/tmp/inputgen_record_outfile";

  pid_t child = fork();
  if (child == 0) {
    if (ptrace(PTRACE_ATTACH, parent_pid, nullptr, nullptr) == -1) {
      perror("ptrace attach");
      abort();
    }
    waitpid(parent_pid, nullptr, 0);

    if (!__ig::storage::dump_memory(parent_pid, outfile, Name, Args)) {
      std::cerr << "Memory dump failed.\n";
      abort();
    }

    ptrace(PTRACE_DETACH, parent_pid, nullptr, nullptr);
    printf("Memory dump finished (child) to '%s'\n", outfile);
    exit(0);
  } else {
    if (waitpid(child, nullptr, 0) == -1) {
      printf("waitpid failed\n");
      exit(1);
    }
    int status = 0;
    if (WIFEXITED(status)) {
      int exit_code = WEXITSTATUS(status);
      if (exit_code == 0) {
        std::cout << "Child exited successfully.\n";
      } else {
        std::cout << "Child exited with code: " << exit_code << "\n";
        exit(1);
      }
    } else {
      std::cout << "Child did not exit normally.\n";
      exit(1);
    }
    printf("Memory dump finished to '%s'.\n", outfile);
    printf("Will now continue executing %s\n", Name);
  }
}

IG_API_ATTRS
void __ig_record_pop(const char *Name, void *Args) {
  printf("Finished execution of %s\n", Name);
  exit(0);
}
