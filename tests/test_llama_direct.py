"""
Direct test of llama-cpp-python without our wrapper.
This isolates whether the issue is in our code or llama-cpp-python itself.
"""
from pathlib import Path

def test_direct_llama():
    """Test llama-cpp-python directly."""
    from llama_cpp import Llama

    model_path = Path("models/selfrag_llama2_7b.Q4_K_M.gguf")
    print(f"Loading model: {model_path}")

    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_batch=512,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("Model loaded!")

    # Simple test
    prompt = """### Instruction:
What is 2+2?

### Response:
"""
    print(f"\nPrompt: {prompt}")
    print("Generating...")

    output = llm(
        prompt,
        max_tokens=50,
        stop=["###"],
        echo=False,
    )

    print(f"Output: {output['choices'][0]['text']}")
    print("\nDirect test PASSED!")


if __name__ == "__main__":
    test_direct_llama()
