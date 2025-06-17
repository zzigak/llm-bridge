import os
import time
from llmbridge import create_llm, choose_provider

# Get API key from environment variable
api_key = os.environ.get('OPENROUTER_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable")

# Set up the interface with caching
choose_provider('openrouter')
model = create_llm('google/gemini-2.5-pro-preview')
model.setup_cache('disk', database_path='cache.db')

def test_caching():
    """Simple test to demonstrate caching clearly."""
    print("SIMPLE CACHING TEST")
    print("=" * 50)
    
    prompt = "Hello, how are you?"
    
    print(f"Prompt: {prompt}")
    print(f"Model: google/gemini-2.5-pro-preview")
    print()
    
    # First call - should be slow and cache the result
    print("1️⃣ FIRST CALL (Will cache):")
    start_time = time.time()
    response1 = model.prompt([prompt], max_tokens=500, temperature=0.5)[0]
    time1 = time.time() - start_time
    
    info1 = model.get_info()
    print(f"   Response: '{response1}'")
    print(f"   Time: {time1:.3f} seconds")
    print(f"   💰 TOTAL RUNNING COST: ${info1['actual_cost']:.6f}")
    print()
    
    # Second call - should be fast and use cache
    print("2️⃣ SECOND CALL (Should use cache):")
    start_time = time.time()
    response2 = model.prompt([prompt], max_tokens=500, temperature=0.5)[0]
    time2 = time.time() - start_time
    
    info2 = model.get_info()
    print(f"   Response: '{response2}'")
    print(f"   Time: {time2:.3f} seconds")
    print(f"   💰 TOTAL RUNNING COST: ${info2['actual_cost']:.6f}")
    print()
    
    # Third call with different temperature - should not use cache
    print("3️⃣ THIRD CALL (Different temperature - won't use cache):")
    start_time = time.time()
    response3 = model.prompt([prompt], max_tokens=500, temperature=0.8)[0]
    time3 = time.time() - start_time
    
    info3 = model.get_info()
    print(f"   Response: '{response3}'")
    print(f"   Time: {time3:.3f} seconds")
    print(f"   💰 TOTAL RUNNING COST: ${info3['actual_cost']:.6f}")
    print()
    
    # Analysis
    print("ANALYSIS:")
    print("=" * 50)
    
    if time2 < time1:
        print(f"✅ CACHING WORKED: Second call ({time2:.3f}s) was faster than first ({time1:.3f}s)")
    else:
        print(f"❌ CACHING FAILED: Second call ({time2:.3f}s) was not faster than first ({time1:.3f}s)")
    
    if response1 == response2:
        print(f"✅ RESPONSES MATCH: Both cached calls returned identical responses")
    else:
        print(f"❌ RESPONSES DIFFER: Cached responses don't match - caching may not be working")
    
    if time3 > time2:
        print(f"✅ DIFFERENT PARAMS: Third call ({time3:.3f}s) was slower than cached call ({time2:.3f}s) - correct behavior")
    else:
        print(f"❌ DIFFERENT PARAMS: Third call was not slower - unexpected behavior")
    
    # Get final stats
    info = model.get_info()
    print(f"\n📊 FINAL STATS:")
    print(f"   Total calls: {info['calls']}")
    print(f"   Total tokens: {info['input_tokens']} input + {info['output_tokens']} output")
    print(f"   💰 TOTAL RUNNING COST: ${info['actual_cost']:.6f}")
    
    # Show the difference between buggy and correct cost
    print(f"\n🐛 BUG DEMONSTRATION:")
    print(f"   Interface 'cost' (buggy): ${info['cost']:.6f}")
    print(f"   Interface 'actual_cost' (correct): ${info['actual_cost']:.6f}")
    print(f"   Difference: ${info['cost'] - info['actual_cost']:.6f}")

if __name__ == "__main__":
    test_caching() 