import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper  # Added numpy_helper import
import os
from numpy.random import randn

# Try to import onnx
try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper  # Added numpy_helper import here too
    onnx_available = True
except ImportError:
    print("ONNX not available, creating a numpy-based dummy model instead")
    onnx_available = False

# Create directory if it doesn't exist
os.makedirs("C:\\Users\\andre\\Desktop\\trae ai mt5 icmarkets\\Data\\Models", exist_ok=True)

if onnx_available:
    # Define model inputs
    price_input = helper.make_tensor_value_info('price_input', TensorProto.FLOAT, [1, 60, 5])
    news_input = helper.make_tensor_value_info('news_input', TensorProto.FLOAT, [1, 60, 1])
    
    # Define model outputs
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])
    
    # Create a constant tensor for weights
    # Change the weights shape to match the reshaped input (300 x 10)
    weights = np.random.randn(300, 10).astype(np.float32)  # 60*5=300 features to 10 outputs
    weights_tensor = numpy_helper.from_array(weights, name='weights')
    
    # Create a node that reshapes the input
    reshape_node = helper.make_node(
        'Reshape',
        inputs=['price_input', 'reshape_shape'],
        outputs=['reshaped_input'],
        name='reshape'
    )
    
    # Create a constant tensor for reshape
    reshape_shape = np.array([1, 60*5]).astype(np.int64)
    reshape_tensor = numpy_helper.from_array(reshape_shape, name='reshape_shape')
    
    # Create a node that does matrix multiplication
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['reshaped_input', 'weights'],
        outputs=['matmul_output'],
        name='matmul'
    )
    
    # Create a node that reshapes the output
    reshape_out_node = helper.make_node(
        'Reshape',
        inputs=['matmul_output', 'reshape_out_shape'],
        outputs=['output'],
        name='reshape_out'
    )
    
    # Create a constant tensor for output reshape
    reshape_out_shape = np.array([1, 10]).astype(np.int64)
    reshape_out_tensor = numpy_helper.from_array(reshape_out_shape, name='reshape_out_shape')
    
    # Create the graph
    graph = helper.make_graph(
        [reshape_node, matmul_node, reshape_out_node],
        'gold_predictor',
        [price_input, news_input],
        [output],
        initializer=[weights_tensor, reshape_tensor, reshape_out_tensor]
    )
    
    # Create the model with opset 11 (widely supported)
    model = helper.make_model(
        graph, 
        producer_name='gold_predictor',
        opset_imports=[helper.make_opsetid("", 11)]  # Use opset 11 for maximum compatibility
    )
    
    # Check model
    onnx.checker.check_model(model)
    
    # Save the model
    onnx.save(model, "C:\\Users\\andre\\Desktop\\trae ai mt5 icmarkets\\Data\\Models\\gold_predictor.onnx")
    print("Dummy ONNX model created successfully with opset 11!")
else:
    # Create a simple numpy-based dummy model
    # This will just be a random weights matrix that we'll save
    # The gold_predictor.py script will need to be modified to handle this
    
    # Create random weights
    weights = randn(5, 10).astype(np.float32)  # 5 features to 10 outputs
    
    # Save as numpy file
    np.save("C:\\Users\\andre\\Desktop\\trae ai mt5 icmarkets\\Data\\Models\\gold_predictor_weights.npy", weights)
    
    # Create a dummy file to indicate this is not a real ONNX model
    with open("C:\\Users\\andre\\Desktop\\trae ai mt5 icmarkets\\Data\\Models\\gold_predictor.onnx", "w") as f:
        f.write("This is a dummy file. Real model is in gold_predictor_weights.npy")
    
    print("Numpy-based dummy model created successfully!")
    print("Note: You'll need to modify gold_predictor.py to use this model instead of ONNX")