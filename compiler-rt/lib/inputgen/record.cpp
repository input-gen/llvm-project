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
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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

static constexpr char InputGenRecordPathEnvVar[] = "INPUTGEN_RECORD_DUMP_PATH";
static constexpr char InputGenRecordDumpFirstNVar[] =
    "INPUTGEN_RECORD_DUMP_FIRST_N";
extern "C" const char __ig_input_dump_path __attribute__((weak));

namespace __ig::storage {

struct MemRegion {
  uintptr_t Start;
  uintptr_t End;
  std::string Perms;
};

std::vector<MemRegion> parse_maps(pid_t Pid) {
  std::vector<MemRegion> Regions;
  std::ifstream Maps("/proc/" + std::to_string(Pid) + "/maps");
  std::string Line;

  while (std::getline(Maps, Line)) {
    std::istringstream iss(Line);
    std::string AddrRange, Perms;
    if (!(iss >> AddrRange >> Perms))
      continue;

    auto dash = AddrRange.find('-');
    uintptr_t Start = std::stoul(AddrRange.substr(0, dash), nullptr, 16);
    uintptr_t End = std::stoul(AddrRange.substr(dash + 1), nullptr, 16);

    if (Perms[0] == 'r') {
      Regions.push_back({Start, End, Perms});
    }
  }

  return Regions;
}

size_t read_all(std::ifstream &MemStream, uintptr_t Addr, char *Buffer,
                size_t Size) {
  MemStream.clear(); // Clear any error flags (EOF, failbit, etc.)
  MemStream.seekg(Addr, std::ios::beg);
  if (!MemStream.good())
    return false;

  size_t total = 0;
  while (total < Size) {
    MemStream.read(Buffer + total, Size - total);
    std::streamsize bytes_read = MemStream.gcount();
    if (bytes_read <= 0)
      break;
    total += bytes_read;
  }

  return total;
}

bool dump_memory(pid_t Pid, const std::string &OutFile, const char *NameC,
                 void *Args) {
  auto Regions = parse_maps(Pid);
  uintptr_t ArgsUint = reinterpret_cast<uintptr_t>(Args);
  std::ofstream OFS(OutFile, std::ios::binary);
  if (!OFS.is_open()) {
    std::cerr << "Failed to open file '" << OutFile
              << "': " << std::strerror(errno) << "\n";
    return false;
  }

  DEFINE_WRITEV(OFS);

  WRITEV(RecordFileMagic);

  std::string Name = NameC;
  uint32_t NameSize = Name.size();
  WRITEV(NameSize);
  OFS.write(Name.data(), NameSize);

  OFS.write(reinterpret_cast<const char *>(&Args), sizeof(Args));

  std::ifstream mem("/proc/" + std::to_string(Pid) + "/mem", std::ios::binary);
  if (!mem) {
    std::cerr << "Failed to open mem file\n";
    return false;
  }

  bool FoundArgs = false;
  for (const auto &Region : Regions) {
    size_t Size = Region.End - Region.Start;
    assert(Size > 0);
    std::vector<char> Buffer(Size);

    size_t ReadBytes = read_all(mem, Region.Start, Buffer.data(), Size);
    if (ReadBytes != Size) {
      std::cerr << "Warning: Partial or failed read at region 0x" << std::hex
                << Region.Start << "-0x" << Region.End << "\n";
    }
    DEBUGF("WRITE IN START 0x%lx SIZE 0x%lx\n", Region.Start,
           Region.End - Region.Start);
    OFS.write(reinterpret_cast<const char *>(&Region.Start),
              sizeof(Region.Start));
    OFS.write(reinterpret_cast<const char *>(&Size), sizeof(Size));
    OFS.write(Buffer.data(), Size);
    if (ArgsUint >= Region.Start && ArgsUint < Region.End) {
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

static std::map<std::string, int> NumInputs;

IG_API_ATTRS
void __ig_record_push(const char *Name, void *Args) {
  printf("Starting recording of %s\n", Name);
  pid_t ParentPid = getpid();

  // TODO this can be put in a global constructor
  static const std::filesystem::path OutFile = [&]() {
    std::string OutFile;
    if (char *DumpPathC = getenv(InputGenRecordPathEnvVar)) {
      OutFile = DumpPathC;
    } else if (&__ig_input_dump_path != nullptr) {
      OutFile = &__ig_input_dump_path;
    } else {
      std::cerr
          << "Input dump path not specified under compilation or runtime. "
             "Please define "
          << InputGenRecordPathEnvVar << "\n";
      abort();
    }

    if (OutFile.empty()) {
      std::cerr << "Input dump path is empty.\n";
      abort();
    }

    std::cerr << "Dumping inputs to " << OutFile << "\n";

    return OutFile;
  }();

  std::filesystem::path ThisOutDir = OutFile / Name;

  std::error_code EC;
  if (!std::filesystem::create_directories(ThisOutDir, EC)) {
    if (EC) {
      std::cerr << "Failed to create dump directory " << ThisOutDir << ": "
                << EC.message() << "\n";
      exit(1);
    }
  }

  auto [CurInput, Existing] = NumInputs.insert({Name, 0});
  std::filesystem::path ThisOutFile =
      ThisOutDir / ("input-" + std::to_string(CurInput->second) + ".inp");
  CurInput->second++;

  pid_t ChildPid = fork();
  if (ChildPid == 0) {
    if (ptrace(PTRACE_ATTACH, ParentPid, nullptr, nullptr) == -1) {
      perror("ptrace attach");
      exit(1);
    }
    waitpid(ParentPid, nullptr, 0);

    if (!__ig::storage::dump_memory(ParentPid, ThisOutFile, Name, Args)) {
      std::cerr << "Memory dump failed.\n";
      exit(1);
    }

    ptrace(PTRACE_DETACH, ParentPid, nullptr, nullptr);
    printf("Memory dump finished (child) to '%s'\n", ThisOutFile.c_str());
    exit(0);
  } else {
    int Status = 0;
    if (waitpid(ChildPid, &Status, 0) == -1) {
      printf("waitpid failed\n");
      exit(1);
    }
    if (WIFEXITED(Status)) {
      int ExitCode = WEXITSTATUS(Status);
      if (ExitCode == 0) {
        std::cout << "Child exited successfully.\n";
      } else {
        std::cout << "Child exited with code: " << ExitCode << "\n";
        exit(1);
      }
    } else {
      std::cout << "Child did not exit normally.\n";
      exit(1);
    }
    printf("Memory dump finished to '%s'.\n", ThisOutFile.c_str());
    printf("Will now continue executing %s\n", Name);
  }
}

IG_API_ATTRS
void __ig_record_pop(const char *Name, void *Args) {
  printf("Finished execution of %s\n", Name);
  static auto DumpFirstN = [&]() -> std::optional<long long> {
    if (char *N = getenv(InputGenRecordDumpFirstNVar))
      return atoll(N) - 1;
    return std::nullopt;
  }();
  if (!DumpFirstN)
    return;
  if (*DumpFirstN == 0)
    exit(0);
  DumpFirstN = *DumpFirstN - 1;
}
