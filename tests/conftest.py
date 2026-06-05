import os
import pytest

# Force mock mode for all tests to avoid rate limits and need for live databases
os.environ["USE_MOCK_LLM"] = "true"
os.environ["USE_MOCK_TOOLS"] = "true"
