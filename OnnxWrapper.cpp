#include <onnxruntime_c_api.h>
#include <vector>
#include <string>
#include <iostream>

// Global session environment
OrtEnv* g_OrtEnv = nullptr;
std::vector<OrtSession*> g_Sessions;

extern "C" {
    __declspec(dllexport) int OrtCreateSession(const char* model_path, int* handle) {
        // Create environment if not already created
        if (!g_OrtEnv) {
            OrtStatus* status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "GoldPredictor", &g_OrtEnv);
            if (status != nullptr) {
                const char* msg = OrtGetErrorMessage(status);
                std::cerr << "Error creating environment: " << msg << std::endl;
                OrtReleaseStatus(status);
                return -1;
            }
        }

        // Create session options
        OrtSessionOptions* session_options;
        OrtStatus* status = OrtCreateSessionOptions(&session_options);
        if (status != nullptr) {
            const char* msg = OrtGetErrorMessage(status);
            std::cerr << "Error creating session options: " << msg << std::endl;
            OrtReleaseStatus(status);
            return -2;
        }

        // Create session
        OrtSession* session;
        status = OrtCreateSession(g_OrtEnv, model_path, session_options, &session);
        OrtReleaseSessionOptions(session_options);
        
        if (status != nullptr) {
            const char* msg = OrtGetErrorMessage(status);
            std::cerr << "Error creating session: " << msg << std::endl;
            OrtReleaseStatus(status);
            return -3;
        }

        // Store session and return handle
        g_Sessions.push_back(session);
        *handle = g_Sessions.size() - 1;
        return 0;
    }

    __declspec(dllexport) int OrtRunInference(int handle, double* input_price, double* input_news, double* output) {
        if (handle < 0 || handle >= g_Sessions.size()) {
            return -1;
        }

        OrtSession* session = g_Sessions[handle];
        
        // Create input tensors
        OrtMemoryInfo* memory_info;
        OrtStatus* status = OrtCreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        if (status != nullptr) {
            OrtReleaseStatus(status);
            return -2;
        }

        // Create price input tensor (shape: [1, 60, 5])
        const int64_t price_dims[] = {1, 60, 5};
        OrtValue* price_tensor;
        status = OrtCreateTensorWithDataAsOrtValue(
            memory_info, input_price, 60 * 5 * sizeof(double),
            price_dims, 3, ONNX_TENSOR_ELEMENT_