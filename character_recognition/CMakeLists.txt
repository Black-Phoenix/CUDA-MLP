set(SOURCE_FILES
    "common.h"
    "common.cu"
    "cpu.h"
    "cpu.cu"
    "naive.h"
    "naive.cu"
    "efficient.h"
    "efficient.cu"
    "thrust.h"
    "thrust.cu"
    )

cuda_add_library(character_recognition
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
