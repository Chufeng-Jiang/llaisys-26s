#pragma once

#include "../tensor/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

enum class Qwen2WorkspaceSlot : std::size_t {
	InputIds = 0,
	PositionIds,

	Hidden0,
	Hidden1,

	AttentionNorm,
	Query,
	Key,
	Value,
	RotatedQuery,
	RotatedKey,
	AttentionOutput,
	AttentionProjection,
	AttentionResidual,

	MlpGate,
	MlpUp,
	MlpActivated,
	MlpOutput,

	Logits,
	MaxIndex,
	MaxValue,

	Count
};

class Qwen2Workspace {
public:
	Qwen2Workspace(
		llaisysDeviceType_t device,
		int device_id
	);

	llaisys::tensor_t get(
		Qwen2WorkspaceSlot slot,
		const std::vector<std::size_t> &shape,
		llaisysDataType_t dtype
	);

	std::vector<std::int64_t> &position_values(
		std::size_t size
	);

private:
	struct Entry {
		llaisys::tensor_t backing;
		std::size_t capacity_elements{0};
		llaisysDataType_t dtype{};
		bool initialized{false};
	};

	static constexpr std::size_t SLOT_COUNT =
		static_cast<std::size_t>(
			Qwen2WorkspaceSlot::Count
		);

	static std::size_t checked_numel(
		const std::vector<std::size_t> &shape
	);

	static std::size_t grow_capacity(
		std::size_t current,
		std::size_t required
	);

	llaisysDeviceType_t device_;
	int device_id_;

	std::array<Entry, SLOT_COUNT> entries_{};
	std::vector<std::int64_t> position_values_;
};
