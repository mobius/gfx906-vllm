#!/usr/bin/env python3
"""
Basic validation test for Qwen3.5 model support.
This test validates that the code structure is correct without requiring GPU.
"""

import sys
import os

def test_imports():
    """Test that all Qwen3.5 modules can be imported."""
    print("Testing Qwen3.5 imports...")
    
    try:
        # Test config imports
        from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
        print("✓ Qwen3_5TextConfig import successful")
        
        from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig
        print("✓ Qwen3_5MoeTextConfig import successful")
        
        # Test config class import
        from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig
        print("✓ Qwen3_5ForConditionalGenerationConfig import successful")
        
        print("\n✅ All Qwen3.5 imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_registry_entries():
    """Test that models are registered in registry."""
    print("\nTesting registry entries...")
    
    try:
        from vllm.model_executor.models.registry import (
            _MULTIMODAL_MODELS,
            _SPECULATIVE_DECODING_MODELS
        )
        
        # Check main models
        assert "Qwen3_5ForConditionalGeneration" in _MULTIMODAL_MODELS
        print("✓ Qwen3_5ForConditionalGeneration registered")
        
        assert "Qwen3_5MoeForConditionalGeneration" in _MULTIMODAL_MODELS
        print("✓ Qwen3_5MoeForConditionalGeneration registered")
        
        # Check speculative decoding models
        assert "Qwen3_5MTP" in _SPECULATIVE_DECODING_MODELS
        print("✓ Qwen3_5MTP registered")
        
        assert "Qwen3_5MoeMTP" in _SPECULATIVE_DECODING_MODELS
        print("✓ Qwen3_5MoeMTP registered")
        
        print("\n✅ All registry entries verified!")
        return True
        
    except Exception as e:
        print(f"\n❌ Registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_structure():
    """Test that model classes are defined."""
    print("\nTesting model class structure...")
    
    try:
        # These imports will test the file structure
        # We don't need to actually instantiate the models
        import vllm.model_executor.models.qwen3_5 as qwen3_5_module
        assert hasattr(qwen3_5_module, 'Qwen3_5ForConditionalGeneration')
        print("✓ Qwen3_5ForConditionalGeneration class exists")
        
        import vllm.model_executor.models.qwen3_5_mtp as qwen3_5_mtp_module
        assert hasattr(qwen3_5_mtp_module, 'Qwen3_5MTP')
        print("✓ Qwen3_5MTP class exists")
        
        import vllm.model_executor.models.qwen3_asr as qwen3_asr_module
        assert hasattr(qwen3_asr_module, 'Qwen3ASRForCausalLM')
        print("✓ Qwen3ASRForCausalLM class exists")
        
        print("\n✅ All model classes verified!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("Qwen3.5 Model Support - Basic Validation Test")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("Registry Test", test_registry_entries()))
    results.append(("Model Structure Test", test_model_structure()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Qwen3.5 support is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
