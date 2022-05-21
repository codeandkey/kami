CXX      = g++
CXXFLAGS = -Werror -Wall -g -I/usr/include/torch/csrc/api/include
LDFLAGS  = -ltorch -ltorch_cpu -lc10

SOURCES = $(wildcard src/*.cpp src/chess/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

TEST_SRC = $(wildcard test/*.cpp)
TESTS    = $(TEST_SRC:.cpp=)

OUTPUT = bin/kami

.PHONY: test build_tests clean

all: $(OUTPUT)

$(OUTPUT): $(OBJECTS)
	mkdir -p bin
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $(OUTPUT)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: build_tests
	$(foreach t, $(TESTS), test/bin/$(notdir $(t)); if [ $? != 0 ]; then echo "$(t) passed" 1>&2; else echo "$(t) failed" 1>&2; fi;)

build_tests:
	@mkdir -p test/bin
	$(foreach t, $(TESTS), $(CXX) $(CXXFLAGS) $(t).cpp src/chess/thc.cpp $(LDFLAGS) -o test/bin/$(notdir $(t));)

clean:
	rm -f $(TESTS) $(OUTPUT) $(OBJECTS)
