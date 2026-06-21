# Makefile — libmanifold.dylib (QSM kernel + Gemma 2 2B inference engine)
#
# Targets:
#   make          — build libmanifold.dylib
#   make test     — quick smoke test via gemma_test.c
#   make clean    — remove build artifacts

CC      = clang
CFLAGS  = -O3 -march=native -ffast-math -Wall -Wextra \
          -Wno-unused-parameter -Wno-unused-function \
          -fPIC
LDFLAGS = -dynamiclib -framework Foundation -lm

SRCS    = manifold.c gemma_infer.c
HEADERS = manifold.h gemma_infer.h gemma_onix.h
OUT     = libmanifold.dylib

.PHONY: all simd test clean

all: simd

simd: CFLAGS += -DMANIFOLD_SIMD
simd: $(OUT)

$(OUT): $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRCS)
	@echo "✅ Built $@"
	@ls -lh $@

test: simd gemma_test.c
	$(CC) $(CFLAGS) -DMANIFOLD_SIMD -o gemma_test gemma_test.c \
	    -L. -lmanifold -Wl,-rpath,. -lm
	@echo "✅ Built gemma_test"
	./gemma_test

clean:
	rm -f $(OUT) gemma_test *.o
