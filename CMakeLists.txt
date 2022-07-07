cmake_minimum_required(VERSION 2.8.11)

project (kami)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#if (MSVC)
#    add_compile_options(/W4 /WX)
#else()
#    add_compile_options(-Wall -Wextra -pedantic -Werror)
#endif()

set(CMAKE_BUILD_TYPE Debug)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET kami 
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:kami>)
endif (MSVC)

add_subdirectory(test)
add_subdirectory(src)
