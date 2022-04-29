CC = g++
CFLAGS = -std=c++17 -Wall -Werror -g
LDFLAGS =

SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:.c=.o)

OUTPUT = kami

.PHONY: clean

all: $(OUTPUT)

$(OUTPUT): $(OBJECTS)
	$(CC) $^ $(LDFLAGS) -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(OUTPUT)
