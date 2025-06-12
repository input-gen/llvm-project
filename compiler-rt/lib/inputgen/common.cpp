#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "common.h"
#include "logging.h"

namespace __ig {

void printAvailableFunctions() {
  ERR("  Available functions:\n");
  for (uint32_t I = 0; I < __ig_num_entry_points; I++)
    ERR("    {}: {}\n", I, __ig_entry_point_names[I]);
}
void printNumAvailableFunctions() {
  ERR("  Num available functions: {}\n", __ig_num_entry_points);
}

IG_API_ATTRS
void __ig_anti_dce(char *ptr, size_t size) {
  if (getenv("VERY_LONG_ENV_NAME_THAT_WILL_LIKELY_NEVER_BE_ACCIDENTALLY_"
             "DEFINED")) {
    for (size_t i = 0; i < size; i++)
      printf("%d", (int)ptr[i]);
    printf("\n");
  }
}

} // namespace __ig
