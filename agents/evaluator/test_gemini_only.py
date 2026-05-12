"""
Quick test to verify Gemini-only setup works.
"""
import os
import logging
from dotenv import load_dotenv

# Load .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test 1: Check API key
print("\n1️⃣ Checking GEMINI_API_KEY...")
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"   ✓ API key found (length: {len(api_key)})")
else:
    print("   ❌ GEMINI_API_KEY not set")
    print("   Set it with: export GEMINI_API_KEY='your_key'")
    exit(1)

# Test 2: Import Gemini client
print("\n2️⃣ Importing GeminiClient...")
try:
    from src.evaluator.llm_client import GeminiClient
    print("   ✓ GeminiClient imported")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    exit(1)

# Test 3: Initialize client
print("\n3️⃣ Initializing Gemini client...")
try:
    client = GeminiClient()
    print("   ✓ Gemini client initialized")
except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    exit(1)

# Test 4: Generate question
print("\n4️⃣ Testing question generation...")
try:
    question = client.generate(
        "Generate a Socratic question about photosynthesis for a high school student.",
        max_tokens=100,
        temperature=0.5
    )
    print(f"   ✓ Question generated:\n   {question[:100]}...")
except Exception as e:
    print(f"   ❌ Generation failed: {e}")
    exit(1)

# Test 5: Initialize evaluator
print("\n5️⃣ Initializing EvaluatorAgent...")
try:
    from src.evaluator.evaluator_agent import EvaluatorAgent
    agent = EvaluatorAgent()
    print("   ✓ EvaluatorAgent initialized")
except Exception as e:
    print(f"   ❌ Agent initialization failed: {e}")
    exit(1)

# Test 6: Start session
print("\n6️⃣ Starting evaluation session...")
try:
    result = agent.start_session(
        task_id="test_001",
        task_title="Photosynthesis",
        task_description="Understanding how plants convert light to energy",
        task_details="Cover light reactions, dark reactions, and the role of chlorophyll",
        key_concepts=["chlorophyll", "ATP", "glucose", "light_reaction", "dark_reaction"],
    )
    print(f"   ✓ Session started: {result['session_id']}")
    print(f"   First question: {result['question'][:80]}...")
except Exception as e:
    print(f"   ❌ Session start failed: {e}")
    exit(1)

print("\n✅ All tests passed! Gemini-only setup is working.\n")