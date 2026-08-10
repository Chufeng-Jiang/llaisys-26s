add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- ============================================================
-- CPU
-- ============================================================

includes("xmake/cpu.lua")


-- ============================================================
-- NVIDIA
-- ============================================================

option("nv-gpu")
	set_default(false)
	set_showmenu(true)
	set_description(
		"Whether to compile implementations for Nvidia GPU"
	)
option_end()


-- ============================================================
-- MetaX
-- ============================================================

option("metax-gpu")
	set_default(false)
	set_showmenu(true)
	set_description(
		"Whether to compile implementations for MetaX GPU"
	)
option_end()


-- Only one GPU backend may be enabled in one build.
if has_config("nv-gpu")
	and has_config("metax-gpu") then
	raise(
		"Only one GPU backend may be enabled per build."
	)
end


if has_config("nv-gpu") then
	add_defines(
		"ENABLE_NVIDIA_API"
	)

	includes(
		"xmake/nvidia.lua"
	)
end


if has_config("metax-gpu") then
	add_defines(
		"ENABLE_METAX_API"
	)

	includes(
		"xmake/metax.lua"
	)
end


-- ============================================================
-- Utils
-- ============================================================

target("llaisys-utils")
	set_kind("static")

	set_languages("cxx17")
	set_warnings("all", "error")

	if not is_plat("windows") then
		add_cxflags(
			"-fPIC",
			"-Wno-unknown-pragmas"
		)
	end

	add_files(
		"src/utils/*.cpp"
	)

	on_install(function (target)
	end)
target_end()


-- ============================================================
-- Device
-- ============================================================

target("llaisys-device")
	set_kind("static")

	add_deps(
		"llaisys-utils"
	)

	add_deps(
		"llaisys-device-cpu"
	)

	if has_config("nv-gpu") then
		add_deps(
			"llaisys-device-nvidia"
		)
	end

	if has_config("metax-gpu") then
		add_deps(
			"llaisys-device-metax"
		)
	end

	set_languages("cxx17")
	set_warnings("all", "error")

	if not is_plat("windows") then
		add_cxflags(
			"-fPIC",
			"-Wno-unknown-pragmas"
		)
	end

	add_files(
		"src/device/*.cpp"
	)

	on_install(function (target)
	end)
target_end()


-- ============================================================
-- Core
-- ============================================================

target("llaisys-core")
	set_kind("static")

	add_deps(
		"llaisys-utils"
	)

	add_deps(
		"llaisys-device"
	)

	set_languages("cxx17")
	set_warnings("all", "error")

	if not is_plat("windows") then
		add_cxflags(
			"-fPIC",
			"-Wno-unknown-pragmas"
		)
	end

	add_files(
		"src/core/*/*.cpp"
	)

	on_install(function (target)
	end)
target_end()


-- ============================================================
-- Tensor
-- ============================================================

target("llaisys-tensor")
	set_kind("static")

	add_deps(
		"llaisys-core"
	)

	set_languages("cxx17")
	set_warnings("all", "error")

	if not is_plat("windows") then
		add_cxflags(
			"-fPIC",
			"-Wno-unknown-pragmas"
		)
	end

	add_files(
		"src/tensor/*.cpp"
	)

	on_install(function (target)
	end)
target_end()


-- ============================================================
-- Operators
-- ============================================================

target("llaisys-ops")
	set_kind("static")

	add_deps(
		"llaisys-ops-cpu"
	)

	if has_config("nv-gpu") then
		add_deps(
			"llaisys-ops-nvidia"
		)
	end

	--
	-- IMPORTANT:
	--
	-- MetaX .maca objects are NOT placed into a separate
	-- llaisys-ops-metax static archive.
	--
	-- They are compiled directly into the final llaisys shared
	-- library below. This avoids an additional:
	--
	--     device object
	--       -> static archive
	--       -> archive merge
	--       -> final shared library
	--
	-- integration boundary.
	--

	set_languages("cxx17")
	set_warnings("all", "error")

	if not is_plat("windows") then
		add_cxflags(
			"-fPIC",
			"-Wno-unknown-pragmas"
		)
	end

	add_files(
		"src/ops/*/*.cpp"
	)

	on_install(function (target)
	end)
target_end()


-- ============================================================
-- Final LLAISYS shared library
-- ============================================================

target("llaisys")
	set_kind("shared")

	add_deps(
		"llaisys-utils"
	)

	add_deps(
		"llaisys-device"
	)

	add_deps(
		"llaisys-core"
	)

	add_deps(
		"llaisys-tensor"
	)

	add_deps(
		"llaisys-ops"
	)

	-- --------------------------------------------------------
	-- NVIDIA
	-- --------------------------------------------------------

	if has_config("nv-gpu") then
		add_deps(
			"llaisys-device-nvidia"
		)

		add_deps(
			"llaisys-ops-nvidia"
		)
	end

	-- --------------------------------------------------------
	-- MetaX
	-- --------------------------------------------------------

	if has_config("metax-gpu") then
		local maca_root =
			os.getenv("MACA_PATH")
			or os.getenv("MACA_HOME")
			or os.getenv("MACA_ROOT")
			or "/opt/maca"

		local maca_library_dir =
			path.join(
				maca_root,
				"lib"
			)

		--
		-- Explicit direct dependency on the MetaX runtime
		-- backend at the final link boundary.
		--
		add_deps(
			"llaisys-device-metax"
		)

		--
		-- Compile MetaX device kernels directly into
		-- libllaisys.so.
		--
		-- The llaisys.maca rule is defined in
		-- xmake/metax.lua.
		--
		add_files(
			"src/ops/*/metax/*.maca",
			{
				rule = "llaisys.maca"
			}
		)

		add_linkdirs(
			maca_library_dir
		)

		--
		-- Resolve MetaX-specific runtime / compiler
		-- dependencies only at the final shared-library
		-- link boundary.
		--
		add_links(
			"runtime_cu",
			"mccompiler",
			"mcruntime",
			"mcblas"
		)

		add_rpathdirs(
			maca_library_dir
		)
	end

	set_languages("cxx17")
	set_warnings("all", "error")

	if not is_plat("windows") then
		add_cxflags(
			"-fPIC",
			"-Wno-unknown-pragmas"
		)
	end

	if is_plat("linux") then
		add_shflags(
			"-Wl,--no-undefined",
			{
				force = true
			}
		)
	end

	add_files(
		"src/llaisys/*.cc"
	)

	add_files(
		"src/llaisys/**/*.cc"
	)

	add_files(
		"src/models/*.cpp"
	)

	set_installdir(".")

	after_install(function (target)
		print(
			"Copying llaisys to "
			.. "python/llaisys/libllaisys/ .."
		)

		if is_plat("windows") then
			os.cp(
				"bin/*.dll",
				"python/llaisys/libllaisys/"
			)
		end

		if is_plat("linux") then
			os.cp(
				"lib/*.so",
				"python/llaisys/libllaisys/"
			)
		end
	end)
target_end()


-- ============================================================
-- Internal regression test:
-- context / device switching
-- ============================================================

target("test-context-device-switch")
	set_kind("binary")

	set_languages("cxx17")
	set_warnings("all", "error")

	add_includedirs(
		"include"
	)

	add_includedirs(
		"src"
	)

	add_deps(
		"llaisys-core"
	)

	add_deps(
		"llaisys-device"
	)

	add_deps(
		"llaisys-device-cpu"
	)

	add_deps(
		"llaisys-utils"
	)

	-- Provides:
	--
	--     llaisysSetContextRuntime()
	--     llaisysGetRuntimeAPI()
	--
	add_files(
		"src/llaisys/runtime.cc",
		"src/llaisys/error.cc"
	)

	add_files(
		"tmp/test_context_device_switch.cpp"
	)

	-- --------------------------------------------------------
	-- NVIDIA test linkage
	-- --------------------------------------------------------

	if has_config("nv-gpu") then
		add_deps(
			"llaisys-device-nvidia"
		)

		add_defines(
			"ENABLE_NVIDIA_API"
		)

		add_syslinks(
			"cublas",
			"cudart"
		)

		on_config(function (target)
			local cuda_root =
				get_config("cuda")

			if not cuda_root then
				raise(
					"CUDA Toolkit directory was not detected"
				)
			end

			local cuda_include_dir =
				path.join(
					cuda_root,
					"include"
				)

			local cuda_library_dir =
				path.join(
					cuda_root,
					"lib64"
				)

			target:add(
				"includedirs",
				cuda_include_dir
			)

			target:add(
				"linkdirs",
				cuda_library_dir
			)

			target:add(
				"rpathdirs",
				cuda_library_dir
			)
		end)
	end

	-- --------------------------------------------------------
	-- MetaX test linkage
	-- --------------------------------------------------------

	if has_config("metax-gpu") then
		local maca_root =
			os.getenv("MACA_PATH")
			or os.getenv("MACA_HOME")
			or os.getenv("MACA_ROOT")
			or "/opt/maca"

		local maca_library_dir =
			path.join(
				maca_root,
				"lib"
			)

		add_deps(
			"llaisys-device-metax"
		)

		add_linkdirs(
			maca_library_dir
		)

		add_links(
			"mcruntime"
		)

		add_rpathdirs(
			maca_library_dir
		)
	end

	-- Internal regression target.
	on_install(function (target)
	end)
target_end()


-- ============================================================
-- Internal regression test:
-- C API error handling
-- ============================================================

target("test-c-api-error")
	set_kind("binary")

	set_languages("cxx17")
	set_warnings("all", "error")

	add_includedirs(
		"include"
	)

	add_includedirs(
		"src"
	)

	add_deps(
		"llaisys-core"
	)

	add_deps(
		"llaisys-device"
	)

	add_deps(
		"llaisys-device-cpu"
	)

	add_deps(
		"llaisys-utils"
	)

	add_files(
		"src/llaisys/runtime.cc",
		"src/llaisys/error.cc",
		"tmp/test_c_api_error.cpp"
	)

	-- --------------------------------------------------------
	-- NVIDIA test linkage
	-- --------------------------------------------------------

	if has_config("nv-gpu") then
		add_deps(
			"llaisys-device-nvidia"
		)

		add_defines(
			"ENABLE_NVIDIA_API"
		)

		add_syslinks(
			"cublas",
			"cudart"
		)

		on_config(function (target)
			local cuda_root =
				get_config("cuda")

			if cuda_root then
				target:add(
					"includedirs",
					path.join(
						cuda_root,
						"include"
					)
				)

				target:add(
					"linkdirs",
					path.join(
						cuda_root,
						"lib64"
					)
				)

				target:add(
					"rpathdirs",
					path.join(
						cuda_root,
						"lib64"
					)
				)
			end
		end)
	end

	-- --------------------------------------------------------
	-- MetaX test linkage
	-- --------------------------------------------------------

	if has_config("metax-gpu") then
		local maca_root =
			os.getenv("MACA_PATH")
			or os.getenv("MACA_HOME")
			or os.getenv("MACA_ROOT")
			or "/opt/maca"

		local maca_library_dir =
			path.join(
				maca_root,
				"lib"
			)

		add_deps(
			"llaisys-device-metax"
		)

		add_linkdirs(
			maca_library_dir
		)

		add_links(
			"mcruntime"
		)

		add_rpathdirs(
			maca_library_dir
		)
	end

	on_install(function (target)
	end)
target_end()