# ==========================================
#  MAKEFILE RANDOM FOREST (SPM PROJECT)
# ==========================================

# FastFlow path
FF_PATH = ../../fastflow

# compiler
CXX = g++

# flags
CXXFLAGS = -std=c++17 -O3 -Wall -Iinclude -fopenmp -pthread -I $(FF_PATH)

# executable target name
TARGET = RandomForest

# list of source files
SRCS = main.cpp \
       src/Data.cpp \
       src/Tree.cpp \
       src/RandomForest.cpp

# target object files
OBJS = $(SRCS:.cpp=.o)

# default rule (if only 'make' is called)
all: $(TARGET)

# link all object files to create the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# rule to compile .cpp files to .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# make clean to remove object files and the executable
clean:
	rm -f $(OBJS) $(TARGET)
	rm -f src/*.o  # Rimuove anche gli oggetti nella sottocartella per sicurezza

# avoid conflict with files named 'all' or 'clean'
.PHONY: all clean