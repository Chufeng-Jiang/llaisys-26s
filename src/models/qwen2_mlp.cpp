#include "qwen2_mlp.hpp"

#include "qwen2_workspace.hpp"

#include "../ops/add/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <stdexcept>

namespace {

Qwen2Workspace &workspace_of(
	LlaisysQwen2Model &model
) {
	if (model.workspace == nullptr) {
		throw std::runtime_error(
			"Qwen2 workspace is not initialized."
		);
	}

	return *model.workspace;
}

} // namespace

void qwen2_mlp_forward(
	LlaisysQwen2Model &model,
	const Qwen2LayerWeights &weights,
	const llaisys::tensor_t &attention_residual,
	std::size_t sequence_length,
	const llaisys::tensor_t &output_hidden
) {
	auto &workspace =
		workspace_of(model);

	const auto dtype =
		model.meta.dtype;

	auto normalized_mlp =
		workspace.get(
			Qwen2WorkspaceSlot::AttentionNorm,
			{
				sequence_length,
				model.meta.hs
			},
			dtype
		);

	llaisys::ops::rms_norm(
		normalized_mlp,
		attention_residual,
		weights.mlp_norm,
		model.meta.epsilon
	);

	auto gate =
		workspace.get(
			Qwen2WorkspaceSlot::MlpGate,
			{
				sequence_length,
				model.meta.di
			},
			dtype
		);

	auto up =
		workspace.get(
			Qwen2WorkspaceSlot::MlpUp,
			{
				sequence_length,
				model.meta.di
			},
			dtype
		);

	llaisys::ops::linear(
		gate,
		normalized_mlp,
		weights.gate_weight,
		nullptr
	);

	llaisys::ops::linear(
		up,
		normalized_mlp,
		weights.up_weight,
		nullptr
	);

	auto activated =
		workspace.get(
			Qwen2WorkspaceSlot::MlpActivated,
			{
				sequence_length,
				model.meta.di
			},
			dtype
		);

	llaisys::ops::swiglu(
		activated,
		gate,
		up
	);

	auto mlp_output =
		workspace.get(
			Qwen2WorkspaceSlot::MlpOutput,
			{
				sequence_length,
				model.meta.hs
			},
			dtype
		);

	llaisys::ops::linear(
		mlp_output,
		activated,
		weights.down_weight,
		nullptr
	);

	llaisys::ops::add(
		output_hidden,
		attention_residual,
		mlp_output
	);
}
