#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc <= 1) {
    printf("not enough args\n");
    return 1;
  }
  const char *infile = argv[1];
  FILE *f = fopen(infile, "rb");
  if (!f) {
    perror("fopen");
    return 1;
  }

  while (1) {
    unsigned long start;
    size_t size;

    if (fread(&start, sizeof(start), 1, f) != 1)
      break;
    if (fread(&size, sizeof(size), 1, f) != 1)
      break;

    void *mapped = mmap((void *)start, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (mapped == MAP_FAILED) {
      perror("mmap");
      return 1;
    }

    if (fread(mapped, 1, size, f) != size) {
      perror("fread data");
      return 1;
    }
  }

  fclose(f);

  printf("Memory restored. Press Enter to continue...\n");
  getchar();
  return 0;
}
