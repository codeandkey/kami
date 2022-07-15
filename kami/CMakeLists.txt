add_subdirectory(chess)

add_executable(kami
  kami.cpp
  nn/nn.cpp
  selfplay.cpp
  evaluate.cpp
  options.cpp
)

target_link_libraries(kami "${TORCH_LIBRARIES}" neocortex thc)
target_precompile_headers(kami PUBLIC nn/nn.h)
set_property(TARGET kami PROPERTY CXX_STANDARD 17)
