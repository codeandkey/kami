add_subdirectory(chess)

add_library(kamicommon
  nn/nn.cpp
  selfplay.cpp
  evaluate.cpp
  options.cpp
)

target_link_libraries(kamicommon "${TORCH_LIBRARIES}" neocortex thc)
#target_precompile_headers(kamicommon PUBLIC nn/nn.h)
set_property(TARGET kamicommon PROPERTY CXX_STANDARD 17)

add_executable(kami kami.cpp)
target_link_libraries(kami kamicommon)
