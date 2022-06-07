cmake_minimum_required(VERSION 2.8.11)

project (kami)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(kami
  src/kami.cpp
  src/chess/thc.cpp
  src/nn/nn.cpp
  src/selfplay.cpp
  src/evaluate.cpp
  src/options.cpp
)

target_link_libraries(kami "${TORCH_LIBRARIES}")
set_property(TARGET kami PROPERTY CXX_STANDARD 17)

#set(CMAKE_BUILD_TYPE Debug)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET kami 
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:kami>)
endif (MSVC)

add_subdirectory(test)
