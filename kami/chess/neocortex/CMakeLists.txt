add_library (neocortex attacks.c board.c zobrist.c position.c types.c)
target_link_libraries(neocortex m)
