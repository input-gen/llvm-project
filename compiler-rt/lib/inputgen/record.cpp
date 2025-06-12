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

bool dump_memory(pid_t pid, const std::string &out_file, void *Args) {
  auto regions = parse_maps(pid);
  std::ofstream out(out_file, std::ios::binary);
  if (!out)
    return false;

  out.write(reinterpret_cast<const char *>(&Args), sizeof(Args));

  std::string mem_path = "/proc/" + std::to_string(pid) + "/mem";
  int mem_fd = open(mem_path.c_str(), O_RDONLY);
  if (mem_fd < 0)
    return false;

  for (const auto &region : regions) {
    size_t size = region.end - region.start;
    std::vector<char> buffer(size);

    if (lseek(mem_fd, region.start, SEEK_SET) == -1)
      continue;
    ssize_t n = read(mem_fd, buffer.data(), size);
    if (n <= 0)
      continue;

    out.write(reinterpret_cast<const char *>(&region.start),
              sizeof(region.start));
    out.write(reinterpret_cast<const char *>(&size), sizeof(size));
    out.write(buffer.data(), n);
  }

  close(mem_fd);
  return true;
}

IG_API_ATTRS
void __ig_record_push(const char *Name, void *Args) {
  printf("PUSH %s\n", Name);
  pid_t parent_pid = getpid();
  const char *outfile = "/home/ivan/tmp/inputgen_record_outfile";

  pid_t child = fork();
  if (child == 0) {
    if (ptrace(PTRACE_ATTACH, parent_pid, nullptr, nullptr) == -1) {
      perror("ptrace attach");
      abort();
    }
    waitpid(parent_pid, nullptr, 0);

    if (!dump_memory(parent_pid, outfile, Args)) {
      std::cerr << "Memory dump failed.\n";
      abort();
    }

    ptrace(PTRACE_DETACH, parent_pid, nullptr, nullptr);
    printf("Memory dump finished (child) to '%s'\n", outfile);
    abort();
  } else {
    waitpid(child, nullptr, 0);
    printf("Memory dump finished to '%s'.\n", outfile);
    abort();
  }
}

IG_API_ATTRS
void __ig_record_pop(const char *Name, void *Args) { printf("POP %s\n", Name); }
