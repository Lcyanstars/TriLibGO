#pragma once

#include "trilibgo/ai/analysis_model.h"

#include <memory>
#include <string>

namespace trilibgo::ai {

/** \brief Check if ONNX Runtime is available (compiled with TRILIBGO_HAS_ONNXRUNTIME). */
[[nodiscard]] bool onnx_runtime_available();

/** \brief Load an ONNX model from the given path. Returns null on failure; error message written to `error`. */
[[nodiscard]] std::shared_ptr<IAnalysisModel> load_onnx_analysis_model(const std::string& model_path, std::string& error);

}  // namespace trilibgo::ai
