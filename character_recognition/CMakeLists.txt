set(SOURCE_FILES
    "common.h"
    "common.cu"
    "mlp.h"
    "mlp.cu"
    )

cuda_add_library(character_recognition
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
