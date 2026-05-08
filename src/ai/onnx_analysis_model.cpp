#include "trilibgo/ai/onnx_analysis_model.h"

#include "trilibgo/ai/feature_encoder.h"

#if defined(TRILIBGO_HAS_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>
#endif

namespace trilibgo::ai {

#if defined(TRILIBGO_HAS_ONNXRUNTIME)
namespace {

class OnnxAnalysisModel final : public IAnalysisModel {
public:
    explicit OnnxAnalysisModel(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "trilibgo"),
          session_options_(),
          session_(env_, std::wstring(model_path.begin(), model_path.end()).c_str(), configured_options()) {}

    [[nodiscard]] std::optional<AnalysisResult> analyze(const trilibgo::core::GameState& state) const override {
        FeatureEncoder encoder;
        const auto planes = encoder.encode(state);
        const auto global_features = encoder.encode_global_features(state);
        if (planes.empty()) {
            return std::nullopt;
        }

        std::vector<float> input;
        input.reserve(planes.size() * planes.front().size());
        for (const auto& plane : planes) {
            input.insert(input.end(), plane.begin(), plane.end());
        }

        std::array<int64_t, 3> shape{
            1,
            static_cast<int64_t>(planes.size()),
            static_cast<int64_t>(planes.front().size())
        };
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), shape.data(), shape.size());
        std::array<int64_t, 2> global_shape{1, static_cast<int64_t>(global_features.size())};
        auto global_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(global_features.data()),
            global_features.size(),
            global_shape.data(),
            global_shape.size()
        );
        std::array<Ort::Value, 2> input_tensors{std::move(tensor), std::move(global_tensor)};

        const char* input_names[] = {"planes", "global_features"};
        const char* output_names[] = {"policy_logits", "value"};
        auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), input_tensors.size(), output_names, 2);
        if (outputs.size() != 2) {
            return std::nullopt;
        }

        auto logits_info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto logits_shape = logits_info.GetShape();
        const std::size_t logits_count = std::accumulate(logits_shape.begin(), logits_shape.end(), std::size_t{1}, std::multiplies<>());
        auto* logits = outputs[0].GetTensorMutableData<float>();
        auto* value = outputs[1].GetTensorMutableData<float>();

        float max_logit = logits[0];
        for (std::size_t i = 1; i < logits_count; ++i) {
            max_logit = std::max(max_logit, logits[i]);
        }
        std::vector<double> exp_logits(logits_count, 0.0);
        double sum = 0.0;
        for (std::size_t i = 0; i < logits_count; ++i) {
            exp_logits[i] = std::exp(static_cast<double>(logits[i] - max_logit));
            sum += exp_logits[i];
        }

        AnalysisResult result;
        result.winrate = std::clamp((static_cast<double>(value[0]) + 1.0) / 2.0, 0.0, 1.0);
        for (std::size_t i = 0; i < logits_count; ++i) {
            result.policy.push_back({static_cast<int>(i), sum > 0.0 ? exp_logits[i] / sum : 0.0});
        }
        result.top_policy = result.policy;
        std::sort(result.top_policy.begin(), result.top_policy.end(), [](const PolicyLogit& lhs, const PolicyLogit& rhs) {
            return lhs.probability > rhs.probability;
        });
        if (result.top_policy.size() > 8) {
            result.top_policy.resize(8);
        }
        return result;
    }

private:
    static Ort::SessionOptions configured_options() {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        return options;
    }

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
};

}  // namespace
#endif

bool onnx_runtime_available() {
#if defined(TRILIBGO_HAS_ONNXRUNTIME)
    return true;
#else
    return false;
#endif
}

std::shared_ptr<IAnalysisModel> load_onnx_analysis_model(const std::string& model_path, std::string& error) {
#if defined(TRILIBGO_HAS_ONNXRUNTIME)
    try {
        error.clear();
        return std::make_shared<OnnxAnalysisModel>(model_path);
    } catch (const std::exception& ex) {
        error = ex.what();
        return {};
    }
#else
    (void)model_path;
    error = "ONNX Runtime is not available in this build.";
    return {};
#endif
}

}  // namespace trilibgo::ai
