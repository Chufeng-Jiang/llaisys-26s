#pragma once

#include "llaisys/runtime.h"

namespace llaisys::device {

const LlaisysRuntimeAPI *getRuntimeAPI(
	llaisysDeviceType_t device_type
);

const LlaisysRuntimeAPI *getUnsupportedRuntimeAPI();

namespace cpu {

const LlaisysRuntimeAPI *getRuntimeAPI();

} // namespace cpu

#ifdef ENABLE_NVIDIA_API

namespace nvidia {

const LlaisysRuntimeAPI *getRuntimeAPI();

} // namespace nvidia

#endif

#ifdef ENABLE_METAX_API

namespace metax {

const LlaisysRuntimeAPI *getRuntimeAPI();

} // namespace metax

#endif

} // namespace llaisys::device