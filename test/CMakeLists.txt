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

add_executable(nntrain
  nntrain.cpp
  ../src/nn/nn.cpp
)

add_executable(selfplay
  selfplay.cpp
  ../src/nn/nn.cpp
  ../src/selfplay.cpp
  ../src/evaluate.cpp
  ../src/chess/thc.cpp
)

add_executable(play
    play.cpp
  ../src/nn/nn.cpp
  ../src/chess/thc.cpp
)

target_link_libraries(nn "${TORCH_LIBRARIES}")
target_link_libraries(nncuda "${TORCH_LIBRARIES}")
target_link_libraries(nndisk "${TORCH_LIBRARIES}")
target_link_libraries(nntrain "${TORCH_LIBRARIES}")
target_link_libraries(selfplay "${TORCH_LIBRARIES}")
target_link_libraries(play "${TORCH_LIBRARIES}")
