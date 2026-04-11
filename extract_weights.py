#!/usr/bin/env python3
"""
Extract weights from saved h5 model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import tensorflow as tf
from tensorflow import keras

# Load model with compile=False to ignore custom loss
print("Loading model...")
model_h5 = Path('models/lstm/lstm_enhanced_focal.h5')

if model_h5.exists():
    try:
        # Compile'ı ignore et
        model = keras.models.load_model(
            str(model_h5),
            compile=False
        )
        print("✅ Model loaded successfully (compile=False)")
        
        # Weights'i kaydede
        weights_path = Path('models/lstm/lstm_enhanced_focal.weights.h5')
        model.save_weights(str(weights_path))
        print(f"✅ Weights saved: {weights_path}")
        
        # Architecture'ı JSON'a kaydede
        import json
        arch_path = Path('models/lstm/lstm_enhanced_focal.json')
        with open(arch_path, 'w') as f:
            json.dump(model.to_json(), f)
        print(f"✅ Architecture saved: {arch_path}")
        
    except Exception as e:
        print(f"❌ Error with compile=False: {e}")
        print("Trying alternative approach...")
        
        # Alternative: just load weights directory
        try:
            import h5py
            with h5py.File(str(model_h5), 'r') as f:
                print("HDF5 file structure:")
                def print_structure(name, obj):
                    print(f"  {name}")
                f.visititems(print_structure)
        except Exception as e2:
            print(f"Error: {e2}")
else:
    print(f"❌ File not found: {model_h5}")
