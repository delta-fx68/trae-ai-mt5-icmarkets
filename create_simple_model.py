import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import os

# Create directory if it doesn't exist
os.makedirs("C:\\Users\\andre\\Desktop\\trae ai mt5 icmarkets\\Data\\Models", exist_ok=True)

# Create a very simple model for XAUUSD (Gold) prediction
# Define model inputs
price_input = helper.make_tensor_value_info('price_input', TensorProto.FLOAT, [1, 60, 5])
news_input = helper.make_tensor_value_info('news_input', TensorProto.FLOAT, [1, 60, 1])

# Define model outputs
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])

# Create a constant tensor with increasing values (simulating price increase)
base_price = 2880.0  # Current XAUUSD price level
forecast = np.array([[base_price + i*5 for i in range(10)]], dtype=np.float32)
constant_tensor = numpy_helper.from_array(forecast, name='constant_output')

# Create a constant node
constant_node = helper.make_node(
    'Constant',
    inputs=[],
    outputs=['output'],
    value=constant_tensor
)

# Create the graph
graph = helper.make_graph(
    [constant_node],
    'gold_predictor',
    [price_input, news_input],
    [output]
)

# Create the model with opset 9 (widely supported)
model = helper.make_model(
    graph, 
    producer_name='gold_predictor',
    opset_imports=[helper.make_opsetid("", 9)]
)

# Save the model
onnx.save(model, "C:\\Users\\andre\\Desktop\\trae ai mt5 icmarkets\\Data\\Models\\gold_predictor.onnx")
print("Simple XAUUSD prediction model created successfully!")