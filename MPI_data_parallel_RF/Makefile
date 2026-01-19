# ==========================================
#  MAKEFILE PER RANDOM FOREST (MPI + OpenMP)
# ==========================================

# 1. Il Compilatore
# Usiamo mpicxx per gestire sia MPI che il C++ sottostante
CXX = mpicxx

# 2. Le opzioni di compilazione (Flags)
# -fopenmp     : Abilita il supporto multi-thread OpenMP
# -std=c++17   : Standard C++
# -O3          : Ottimizzazioni di alto livello
# -march=native: Sfrutta le istruzioni specifiche della CPU del nodo
# -Wall        : Avvisi del compilatore
# -Iinclude    : Cerca gli header nella cartella include
CXXFLAGS = -std=c++17 -O3 -Wall -Iinclude -fopenmp -pthread

# 3. Nome dell'eseguibile finale
TARGET = RandomForest_hybrid

# 4. Lista dei file sorgente (.cpp)
SRCS = main.cpp \
       src/Data.cpp \
       src/Tree.cpp \
       src/RandomForest.cpp

# 5. File Oggetto
OBJS = $(SRCS:.cpp=.o)

# ==========================================
#  REGOLE
# ==========================================

all: $(TARGET)

# Linking: mpicxx gestisce l'unione di MPI e OpenMP
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compilazione dei singoli file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Pulizia
clean:
	rm -f $(OBJS) $(TARGET)
	rm -f src/*.o

.PHONY: all clean