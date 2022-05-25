add_executable(bench
  bench.cpp
  ../src/chess/thc.cpp
)

add_executable(mcts
  mcts.cpp
  ../src/chess/thc.cpp
)

add_executable(nn
  nn.cpp
  ../src/nn/nn.cpp
)

add_executable(nncuda
  nncuda.cpp
  ../src/nn/nn.cpp
)

add_executable(nndisk
  nndisk.cpp
  ../src/nn/nn.cpp
)

target_link_libraries(nn "${TORCH_LIBRARIES}")
target_link_libraries(nncuda "${TORCH_LIBRARIES}")
target_link_libraries(nndisk "${TORCH_LIBRARIES}")

#set_property(TARGET kami PROPERTY CXX_STANDARD 14)
