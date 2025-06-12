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
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <unistd.h>

#include "common.h"

int dump_maps_and_mem(pid_t target, const char *out_file, const char *Args) {
  char maps_path[64], mem_path[64];
  snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", target);
  snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", target);

  FILE *maps = fopen(maps_path, "r");
  if (!maps)
    return perror("fopen maps"), -1;

  int mem_fd = open(mem_path, O_RDONLY);
  if (mem_fd < 0) {
    fclose(maps);
    return perror("open mem"), -1;
  }

  FILE *out = fopen(out_file, "wb");
  if (!out) {
    perror("fopen out");
    fclose(maps);
    close(mem_fd);
    return -1;
  }

  char line[256];
  while (fgets(line, sizeof(line), maps)) {
    unsigned long start, end;
    char perms[5];
    if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3)
      continue;
    if (perms[0] != 'r')
      continue; // read-only or no read? skip

    size_t size = end - start;
    void *buf = malloc(size);
    if (!buf)
      continue;

    if (lseek(mem_fd, start, SEEK_SET) == -1) {
      free(buf);
      continue;
    }
    ssize_t n = read(mem_fd, buf, size);
    if (n > 0) {
      fwrite(&start, sizeof(start), 1, out);
      fwrite(&size, sizeof(size), 1, out);
      fwrite(buf, 1, n, out);
    }
    free(buf);
  }

  fclose(maps);
  close(mem_fd);
  fclose(out);
  return 0;
}

IG_API_ATTRS
void __ig_record_push(const char *Name, void *Args) {
  printf("PUSH %s\n", Name);
  pid_t parent_pid = getpid();
  const char *outfile = "/home/ivan/tmp/inputgen_record_outfile";

  pid_t child = fork();
  if (child < 0) {
    perror("fork");
    abort();
  }

  if (child == 0) {
    // Child process: ptrace parent, dump, detach
    if (ptrace(PTRACE_ATTACH, parent_pid, NULL, NULL) == -1) {
      perror("ptrace attach");
      abort();
    }
    waitpid(parent_pid, NULL, 0);

    int ret = dump_maps_and_mem(parent_pid, outfile, Args);
    ptrace(PTRACE_DETACH, parent_pid, NULL, NULL);
    printf("Memory dump finished (child) to '%s' status %d.\n", outfile, ret);
    abort();
  } else {
    // Parent just waits
    waitpid(child, NULL, 0);
    printf("Memory dump finished to '%s'.\n", outfile);
    abort();
  }
}

IG_API_ATTRS
void __ig_record_pop(const char *Name, void *Args) { printf("POP %s\n", Name); }
