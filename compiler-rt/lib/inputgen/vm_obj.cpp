#include "vm_obj.h"
#include "vm_storage.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>

using namespace __ig;

ObjectManager::~ObjectManager() {}

void *ObjectManager::getObj(uint32_t Seed) { return add(8, Seed); }

void ObjectManager::reset() {
  UserObjSmall.reset();
  UserObjLarge.reset();
  RTObjs.reset();
  FVM.reset();
  std::set<char *> ArgMemPtrs;
}

void ObjectManager::saveInput(uint32_t EntryNo, uint32_t InputIdx,
                              uint32_t ExitCode) {
  INPUTGEN_DEBUG({
    if (getenv("PRINT_RUNTIME_OBJECTS")) {
      printf("\n\nRuntime objects (%u):\n", RTObjs.TableEntryCnt);
      for (uint32_t I = 0; I < RTObjs.TableEntryCnt; ++I) {
        RTObjs.Table[I].printStats();
      }
    }
  });

  storage::StorageManager SM;

  for (uint32_t I = 0, E = RTObjs.TableEntryCnt; I != E; ++I) {
    SM.encode(*this, I, RTObjs.Table[I]);
  }

  std::string OutputName = ProgramName + "." + std::to_string(EntryNo) + "." +
                           std::to_string(InputIdx) + "." +
                           std::to_string(ExitCode) + ".inp";
  std::ofstream OFS(OutputName, std::ios_base::out | std::ios_base::binary);
  SM.write(OFS);
}

std::function<void(uint32_t)> __ig::ErrorFn;
void __ig::error(uint32_t ErrorCode) {
  if (ErrorFn)
    ErrorFn(ErrorCode);
  else
    printf("Encountered error %u but no stop function available\n", ErrorCode);
  std::terminate();
}
