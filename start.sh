#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting RAG Application...${NC}"

# Kill any existing processes
pkill -f "uvicorn main:app" 2>/dev/null
pkill -f "http.server 3000" 2>/dev/null

# Start backend
echo -e "${BLUE}Starting backend server...${NC}"
cd backend
uvicorn main:app --host 0.0.0.0 --port 7860 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo -e "${BLUE}Starting frontend server...${NC}"
cd frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}✅ Application started!${NC}"
echo -e "${GREEN}📱 Frontend: http://localhost:3000${NC}"
echo -e "${GREEN}⚙️  Backend API: http://localhost:7860${NC}"
echo -e "${GREEN}📚 API Docs: http://localhost:7860/docs${NC}"
echo ""
echo "Press Ctrl+C to stop all servers"

# Handle Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo -e '\n${GREEN}✅ Servers stopped${NC}'; exit" INT
wait