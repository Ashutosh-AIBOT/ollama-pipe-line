#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Starting RAG Application${NC}"
echo -e "${GREEN}================================${NC}"

# Kill any existing processes on our ports
echo -e "${BLUE}Cleaning up existing processes...${NC}"
pkill -f "uvicorn main:app" 2>/dev/null
pkill -f "http.server 3000" 2>/dev/null
pkill -f "ollama serve" 2>/dev/null
sleep 2

# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Ollama is not installed. Please install it first.${NC}"
    exit 1
fi

# Start Ollama (if not already running)
echo -e "${BLUE}Starting Ollama...${NC}"
ollama serve &
OLLAMA_PID=$!
sleep 3

# Pull the model if needed
echo -e "${BLUE}Ensuring phi3:mini model is available...${NC}"
ollama pull phi3:mini

# Start backend
echo -e "${BLUE}Starting backend server...${NC}"
cd backend
# Install dependencies if needed
pip install fastapi uvicorn ollama python-dotenv 2>/dev/null
uvicorn main:app --host 0.0.0.0 --port 7860 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo -e "${BLUE}Waiting for backend to initialize...${NC}"
sleep 5

# Start frontend
echo -e "${BLUE}Starting frontend server...${NC}"
cd frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✅ Application started successfully!${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}📱 Frontend: ${BLUE}http://localhost:3000${NC}"
echo -e "${GREEN}⚙️  Backend API: ${BLUE}http://localhost:7860${NC}"
echo -e "${GREEN}📚 API Docs: ${BLUE}http://localhost:7860/docs${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Press Ctrl+C to stop all servers"

# Handle Ctrl+C to cleanup
trap 'echo -e "\n${BLUE}Stopping servers...${NC}"; \
      kill $BACKEND_PID $FRONTEND_PID $OLLAMA_PID 2>/dev/null; \
      echo -e "${GREEN}✅ Servers stopped${NC}"; \
      exit' INT

# Keep the script running
wait
