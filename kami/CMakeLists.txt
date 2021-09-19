add_executable (kami kami.cpp model.cpp)
target_link_libraries (kami "${TORCH_LIBRARIES}")
set_property (TARGET kami PROPERTY CXX_STANDARD 14)
