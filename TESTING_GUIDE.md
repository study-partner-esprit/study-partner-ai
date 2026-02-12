# Study Partner AI - Full Stack Testing Guide

This guide will help you test the complete AI integration stack: course ingestion, planning, live coaching, and signal monitoring.

## Prerequisites

- Python 3.12+ with Poetry
- Node.js 18+
- MongoDB running locally or accessible
- Terminal/Command line access

## Setup Steps

### 1. Start MongoDB
```bash
# Make sure MongoDB is running
# If using Docker:
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Or start your local MongoDB service
sudo systemctl start mongod
```

### 2. Start AI Backend (FastAPI)
```bash
cd study-partner-ai

# Install dependencies if not already done
poetry install

# Start the AI API service on port 8000
poetry run python -m services.api.main

# Server will be available at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

### 3. Start Node.js Backend (Optional - for auth/sessions)
```bash
cd study-partner-api

# Install dependencies
npm install

# Start all microservices
npm run dev

# Services will run on ports 3001-3003
```

### 4. Start React Frontend
```bash
cd study-partner-web

# Install dependencies
npm install

# Create .env file with:
# VITE_API_URL=http://localhost:3000
# VITE_AI_API_URL=http://localhost:8000

# Start development server
npm run dev

# Frontend will be available at: http://localhost:5173
```

## Testing the AI Features

### 1. Course Upload & Ingestion
1. Navigate to **http://localhost:5173/courses**
2. Enter a course title (e.g., "Linear Algebra")
3. Upload PDF files or text documents
4. Wait for processing (runs in background)
5. Verify: Check MongoDB `course_documents` collection

**What it tests:**
- PDF extraction (PyMuPDF)
- OCR fallback (Tesseract)
- Section detection & parsing
- Subtopic chunking & tokenization
- MongoDB storage

### 2. Study Plan Creation
1. Navigate to **http://localhost:5173/planner**
2. Enter a learning goal (e.g., "Master eigenvalues and eigenvectors")
3. Set available time (e.g., 120 minutes)
4. Optionally select a course
5. Click "Generate Study Plan"
6. Review the generated atomic tasks
7. Click "Start Study Session"

**What it tests:**
- RAG-based course content retrieval
- LLM decomposition (Google Gemini)
- Task graph creation with prerequisites
- Time allocation & difficulty estimation
- Clarification rules & feasibility checks
- Schedule creation in MongoDB

### 3. Live Study Session
1. Navigate to **http://localhost:5173/study-session**
2. Click "Start Study Session"
3. Observe real-time monitoring:
   - **Focus State**: Focused/Drifting/Lost (ML-based)
   - **Fatigue Level**: Alert/Moderate/High/Critical (MediaPipe)
   - **Signal History**: Live chart updates every 10 seconds
4. Watch for Coach popup recommendations (every 30 seconds)
5. Test different scenarios:
   - Enable "Do Not Disturb" → Coach should remain silent
   - Ignore multiple suggestions → Coach respects autonomy
   - Critical fatigue detected → Coach overrides focus

**What it tests:**
- Signal Processing Service (focus & fatigue adapters)
- AI Orchestrator (task + signal aggregation)
- Coach Agent (rule engine + LLM fallback)
- Schedule Orchestrator (automatic rescheduling)
- Real-time frontend updates

### 4. Coach Decision Logic
The coach follows this priority:
1. **Do Not Disturb** → Always silence
2. **Multiple Ignores** (3+) → Silence (respects autonomy)
3. **Critical Fatigue** → Force break (safety first!)
4. **Deep Focus** → Silence (never interrupt flow)
5. **High Fatigue** → Suggest break
6. **Moderate States** → LLM decides (context-aware)

### 5. Schedule Adaptation
When coach suggests a break:
- Schedule Orchestrator inserts break session
- Subsequent tasks automatically shift
- MongoDB `task_scheduling` collection updates
- Frontend can fetch updated schedule

## API Endpoints Reference

### Course Ingestion
- `POST /api/ai/courses/ingest` - Upload PDFs
- `GET /api/ai/courses` - List courses
- `GET /api/ai/courses/{id}` - Get course details

### Planning
- `POST /api/ai/planner/create-plan` - Generate study plan
- `GET /api/ai/planner/plans/{user_id}` - Get user plans

### Coaching
- `POST /api/ai/coach/decision` - Get real-time coaching
- `GET /api/ai/coach/history/{user_id}` - Coach history

### Signal Processing
- `GET /api/ai/signals/current/{user_id}` - Current focus/fatigue
- `GET /api/ai/signals/history/{user_id}` - Signal history
- `POST /api/ai/signals/process` - Trigger signal processing

### Health Check
- `GET /health` - Check all services status

## MongoDB Collections

After testing, verify data in MongoDB:
```bash
mongo

use study_partner

# Course data
db.course_documents.findOne()

# Study plans
db.study_plans.find().pretty()

# Task scheduling
db.task_scheduling.find().pretty()

# Signal snapshots
db.signals.find().sort({timestamp: -1}).limit(5)

# Schedule history (adaptations)
db.schedule_history.find().pretty()
```

## Troubleshooting

### AI Backend won't start
- Check Python version: `python --version` (need 3.12+)
- Install missing dependencies: `poetry install`
- Verify MongoDB connection: `mongodb://localhost:27017`

### Frontend can't connect
- Check `.env` file in `study-partner-web/`
- Verify CORS settings in FastAPI
- Check browser console for errors

### Coach never appears
- Verify signals are being generated (check `/api/ai/signals/current/{user_id}`)
- Check coach decision endpoint directly in browser/Postman
- Look for rule engine debug output in AI backend logs

### Focus/Fatigue always mock values
- Models need to be loaded (check `ML/focus/outputs/` and `ML/fatigue-merged/`)
- Without real video data, adapters use mock signals
- To test with real data: integrate webcam feed (future enhancement)

## Expected Behavior

### Normal Flow
1. **Initial State**: Alert fatigue, drifting focus
2. **After 30s**: Coach suggests focus technique
3. **User accepts**: Task time extended
4. **Focus improves**: Coach stays silent
5. **Fatigue builds**: Coach suggests short break
6. **User continues**: Ignore count increases
7. **Critical fatigue**: Coach overrides, forces break
8. **Schedule adapts**: Tasks rescheduled automatically

### Performance Metrics
- Course ingestion: 10-30s per PDF (depends on size)
- Study plan generation: 5-15s (LLM latency)
- Signal processing: <100ms per frame
- Coach decision: 200-500ms (with LLM) or <50ms (rules only)
- Frontend updates: Real-time (WebSocket would be better)

## Next Steps / Enhancements

1. **Webcam Integration**: Real video feed for accurate focus/fatigue
2. **WebSocket**: Replace polling with bidirectional streaming
3. **PWA**: Offline support & mobile app
4. **Analytics Dashboard**: Historical trends & insights
5. **Social Features**: Study with friends, leaderboards
6. **Voice Coaching**: Text-to-speech recommendations
7. **Pomodoro Integration**: Structured break timing
8. **Export Plans**: PDF/calendar sync

## Success Criteria

✅ Course uploads and processes successfully  
✅ Study plans generate with valid task graphs  
✅ Signals update in real-time during session  
✅ Coach provides context-aware recommendations  
✅ Critical fatigue overrides focus (safety)  
✅ Schedule adapts when breaks inserted  
✅ Frontend displays all data correctly  

## Support

If something doesn't work:
1. Check backend logs (FastAPI terminal)
2. Check frontend console (browser DevTools)
3. Verify MongoDB collections
4. Test API endpoints directly at http://localhost:8000/docs
5. Review this guide's troubleshooting section

Happy studying! 🚀📚
