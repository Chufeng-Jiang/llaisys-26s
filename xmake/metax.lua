-- ============================================================
-- MetaX paths
-- ============================================================

local MACA_ROOT =
	os.getenv("MACA_PATH")
	or os.getenv("MACA_HOME")
	or os.getenv("MACA_ROOT")
	or "/opt/maca"

local MACA_INCLUDE =
	path.join(
		MACA_ROOT,
		"include"
	)

local CUDA_BRIDGE_INCLUDE =
	path.join(
		MACA_ROOT,
		"tools/cu-bridge/include"
	)

local MACA_LIBRARY =
	path.join(
		MACA_ROOT,
		"lib"
	)

local MXCC =
	path.join(
		MACA_ROOT,
		"mxgpu_llvm/bin/mxcc"
	)


-- ============================================================
-- MetaX kernel compilation rule
-- ============================================================

rule("llaisys.maca")
	set_extensions(".maca")

	on_build_file(function (
		target,
		sourcefile
	)
		if not os.isfile(MXCC) then
			raise(
				"MetaX mxcc compiler not found: "
				.. MXCC
			)
		end

		if not os.isdir(MACA_INCLUDE) then
			raise(
				"MetaX MACA include directory not found: "
				.. MACA_INCLUDE
			)
		end

		local objectfile =
			target:objectfile(
				sourcefile
			)

		os.mkdir(
			path.directory(
				objectfile
			)
		)

		local args = {
			"-x",
			"maca",

			"-c",
			sourcefile,

			"-o",
			objectfile,

			"-std=c++17",

			"-O3",

			"-fPIC",

			"-offload-arch",
			"native",

			"--maca-path="
				.. MACA_ROOT,

			"-I"
				.. CUDA_BRIDGE_INCLUDE,

			"-I"
				.. MACA_INCLUDE
		}

		--
		-- Propagate include paths from the final target.
		--
		local includedirs =
			target:get("includedirs")

		if includedirs then
			for _, includedir in ipairs(
				includedirs
			) do
				table.insert(
					args,
					"-I"
						.. includedir
				)
			end
		end

		--
		-- Propagate target definitions, including
		-- ENABLE_METAX_API.
		--
		local defines =
			target:get("defines")

		if defines then
			for _, define in ipairs(
				defines
			) do
				table.insert(
					args,
					"-D"
						.. define
				)
			end
		end

		cprint(
			"${cyan}MetaX compiling: %s",
			sourcefile
		)

		os.execv(
			MXCC,
			args
		)

		--
		-- Register mxcc-generated object with the target.
		--
		table.insert(
			target:objectfiles(),
			objectfile
		)
	end)
rule_end()


-- ============================================================
-- MetaX Runtime backend
-- ============================================================

target("llaisys-device-metax")
	set_kind("static")

	add_deps(
		"llaisys-utils"
	)

	set_languages("cxx17")
	set_warnings("all", "error")

	if not is_plat("windows") then
		add_cxflags(
			"-fPIC",
			"-Wno-unknown-pragmas"
		)
	end

	add_includedirs(
		MACA_INCLUDE
	)

	add_files(
		"../src/device/metax/*.cpp"
	)

	on_config(function (target)
		if not os.isdir(MACA_INCLUDE) then
			raise(
				"MetaX MACA include directory not found: "
					.. MACA_INCLUDE
			)
		end

		if not os.isdir(MACA_LIBRARY) then
			raise(
				"MetaX MACA library directory not found: "
					.. MACA_LIBRARY
			)
		end

		if not os.isfile(MXCC) then
			raise(
				"MetaX mxcc compiler not found: "
					.. MXCC
			)
		end

		cprint(
			"${cyan}MetaX MACA root: %s",
			MACA_ROOT
		)

		cprint(
			"${cyan}MetaX MACA headers: %s",
			MACA_INCLUDE
		)

		cprint(
			"${cyan}MetaX MACA libraries: %s",
			MACA_LIBRARY
		)

		cprint(
			"${cyan}MetaX compiler: %s",
			MXCC
		)
	end)

	on_install(function (target)
	end)
target_end()