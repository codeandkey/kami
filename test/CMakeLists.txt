add_executable(bench
  bench.cpp
  ../kami/options.cpp
)

add_executable(mcts
  mcts.cpp
  ../kami/options.cpp
)

add_executable(nn
  nn.cpp
  ../kami/nn/nn.cpp
  ../kami/options.cpp
)

add_executable(nncuda
  nncuda.cpp
  ../kami/nn/nn.cpp
  ../kami/options.cpp
)

add_executable(nndisk
  nndisk.cpp
  ../kami/nn/nn.cpp
  ../kami/options.cpp
)

add_executable(nntrain
  nntrain.cpp
  ../kami/nn/nn.cpp
  ../kami/options.cpp
)

add_executable(selfplay
  selfplay.cpp
  ../kami/nn/nn.cpp
  ../kami/selfplay.cpp
  ../kami/evaluate.cpp
  ../kami/options.cpp
)

add_executable(play
    play.cpp
  ../kami/nn/nn.cpp
  ../kami/options.cpp
)

add_executable(encoding
    encoding.cpp
)

target_link_libraries(bench neocortex thc)
target_link_libraries(encoding neocortex thc)
target_link_libraries(mcts neocortex thc)
target_link_libraries(nn "${TORCH_LIBRARIES}" neocortex thc)
target_link_libraries(nncuda "${TORCH_LIBRARIES}" neocortex thc)
target_link_libraries(nndisk "${TORCH_LIBRARIES}" neocortex thc)
target_link_libraries(nntrain "${TORCH_LIBRARIES}" neocortex thc)
target_link_libraries(selfplay "${TORCH_LIBRARIES}" neocortex thc)
target_link_libraries(play "${TORCH_LIBRARIES}" neocortex thc)
