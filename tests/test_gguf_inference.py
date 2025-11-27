"""
Test script for SelfRAGGGUFInference to debug llama_decode returned -3 error.
Run with: python test_gguf_inference.py
"""
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_inference():
    """Test basic inference without retrieval."""
    print("=" * 60)
    print("TEST 1: Basic inference (single call)")
    print("=" * 60)

    from src.self_rag.gguf_inference import SelfRAGGGUFInference

    model_path = Path("models/selfrag_llama2_7b.Q4_K_M.gguf")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return False

    print(f"Loading model from {model_path}...")
    inference = SelfRAGGGUFInference(
        model_path=str(model_path),
        n_ctx=2048,
        n_gpu_layers=-1,
    )

    print("\nGenerating response...")
    result = inference.generate(
        "What is 2+2?",
        passage="2+2 equals 4. This is basic arithmetic."
    )

    print(f"Answer: {result.answer}")
    print(f"IsRel: {result.isrel}")
    print(f"IsSup: {result.issup}")
    print(f"IsUse: {result.isuse}")
    print("TEST 1 PASSED\n")
    return True


def test_multiple_calls():
    """Test multiple inference calls in a loop (the failing case)."""
    print("=" * 60)
    print("TEST 2: Multiple calls in loop (5 iterations)")
    print("=" * 60)

    from src.self_rag.gguf_inference import SelfRAGGGUFInference

    model_path = Path("models/selfrag_llama2_7b.Q4_K_M.gguf")

    print(f"Loading model from {model_path}...")
    inference = SelfRAGGGUFInference(
        model_path=str(model_path),
        n_ctx=2048,
        n_gpu_layers=-1,
    )

    questions = [
        ("What is negligence?", "Negligence is a failure to exercise reasonable care."),
        ("What is a contract?", "A contract is a legally binding agreement between parties."),
        ("What is consideration?", "Consideration is something of value exchanged in a contract."),
        ("What is a tort?", "A tort is a civil wrong that causes harm to another person."),
        ("What is liability?", "Liability refers to legal responsibility for one's actions."),
    ]

    for i, (question, passage) in enumerate(questions, 1):
        print(f"\n--- Iteration {i}/5 ---")
        print(f"Question: {question}")

        try:
            result = inference.generate(question, passage=passage)
            print(f"Answer: {result.answer[:100]}...")
            print(f"Tokens: IsRel={result.isrel}, IsSup={result.issup}")
        except Exception as e:
            print(f"ERROR on iteration {i}: {e}")
            return False

    print("\nTEST 2 PASSED\n")
    return True


def test_longer_loop():
    """Test 20 iterations to stress test memory."""
    print("=" * 60)
    print("TEST 3: Stress test (20 iterations)")
    print("=" * 60)

    from src.self_rag.gguf_inference import SelfRAGGGUFInference

    model_path = Path("models/selfrag_llama2_7b.Q4_K_M.gguf")

    print(f"Loading model from {model_path}...")
    inference = SelfRAGGGUFInference(
        model_path=str(model_path),
        n_ctx=2048,
        n_gpu_layers=-1,
    )

    passage = "The answer to this question can be found in legal documents."

    for i in range(1, 21):
        try:
            result = inference.generate(f"Question number {i}?", passage=passage)
            print(f"Iteration {i:2d}: OK - {result.answer[:50]}...")
        except Exception as e:
            print(f"ERROR on iteration {i}: {e}")
            return False

    print("\nTEST 3 PASSED\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GGUF Inference Test Suite")
    print("=" * 60 + "\n")

    # Run tests
    tests = [
        ("Basic inference", test_basic_inference),
        ("Multiple calls", test_multiple_calls),
        ("Stress test", test_longer_loop),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"FATAL ERROR in {name}: {e}")
            results.append((name, False))
            break

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
