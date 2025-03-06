import torch

def test_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Check if cuDNN is available
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        if torch.backends.cudnn.enabled:
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        
        # Create a test tensor on GPU
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = torch.tensor([4.0, 5.0, 6.0]).cuda()
        z = x + y
        print(f"Test GPU computation: {x} + {y} = {z}")
        print(f"Tensor device: {z.device}")
    else:
        print("CUDA is not available. Using CPU instead.")

if __name__ == "__main__":
    test_cuda()