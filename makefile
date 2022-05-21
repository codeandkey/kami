CXX      = g++
CXXFLAGS = -Werror -Wall -g
LDFLAGS  =

TEST_SRC = $(wildcard test/*.cpp)
TESTS    = $(TEST_SRC:.cpp=)

.PHONY: test build_tests clean

QUIET = hello

ifdef ($(QUIET))
	TESTEXT = 2>&1
endif

test: build_tests
	$(foreach t, $(TESTS), test/bin/$(notdir $(t)) $(TESTEXT); if [ $? != 0 ]; then echo "$(t) passed"; else echo "$(t) failed"; fi;)

build_tests:
	@mkdir -p test/bin
	$(foreach t, $(TESTS), $(CXX) $(CXXFLAGS) $(t).cpp src/chess/thc.cpp $(LDFLAGS) -o test/bin/$(notdir $(t));)

clean:
	rm -f $(TESTS)
