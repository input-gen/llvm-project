#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>

namespace __ig {

#ifndef NDEBUG
#define INPUTGEN_DEBUG(X)                                                      \
  do {                                                                         \
    X;                                                                         \
  } while (0)
#else
#define INPUTGEN_DEBUG(X)
#endif

#define INFOF(...) fprintf(stdout, __VA_ARGS__)
#define WARNF(...) fprintf(stderr, __VA_ARGS__)
#define ERRF(...) fprintf(stderr, __VA_ARGS__)
#define VERBOSE(...) INPUTGEN_DEBUG(fprintf(stdout, __VA_ARGS__))
#define DEBUGF(...) INPUTGEN_DEBUG(fprintf(stderr, __VA_ARGS__))

} // namespace __ig

#endif // LOGGING_H
