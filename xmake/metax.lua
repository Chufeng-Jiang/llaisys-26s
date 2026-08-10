-- ============================================================
-- MetaX Runtime
-- ============================================================

target("llaisys-device-metax")
	set_kind("static")

	set_languages("cxx17")
	set_warnings("all", "error")

	if not is_plat("windows") then
		add_cxflags(
			"-fPIC",
			"-Wno-unknown-pragmas"
		)
	end

	local maca_path =
		os.getenv("MACA_PATH")
		or "/opt/maca"

	local maca_include_dir =
		path.join(
			maca_path,
			"include"
		)

	local maca_library_dir =
		path.join(
			maca_path,
			"lib"
		)

	add_includedirs(
		maca_include_dir,
		{public = true}
	)

	add_linkdirs(
		maca_library_dir,
		{public = true}
	)

	add_links(
		"mcruntime",
		{public = true}
	)

	add_rpathdirs(
		maca_library_dir,
		{public = true}
	)

	add_files(
		"../src/device/metax/*.cpp"
	)

	on_config(function (target)
		if not os.isdir(maca_include_dir) then
			raise(
				"MetaX MACA include directory not found: "
					.. maca_include_dir
			)
		end

		if not os.isdir(maca_library_dir) then
			raise(
				"MetaX MACA library directory not found: "
					.. maca_library_dir
			)
		end

		cprint(
			"${cyan}MetaX MACA headers: %s",
			maca_include_dir
		)

		cprint(
			"${cyan}MetaX MACA libraries: %s",
			maca_library_dir
		)
	end)

	on_install(function (target)
	end)
target_end()

-- ============================================================
-- MetaX kernel compilation rule
-- ============================================================

rule("metax.kernel")
	set_extensions(".maca")

	on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
		local maca_path =
			os.getenv("MACA_PATH")

		if not maca_path or #maca_path == 0 then
			raise(
				"MACA_PATH is not set"
			)
		end

		local mxcc =
			path.join(
				maca_path,
				"mxgpu_llvm",
				"bin",
				"mxcc"
			)

		local objectfile =
			target:objectfile(
				sourcefile
			)

		table.insert(
			target:objectfiles(),
			objectfile
		)

		batchcmds:show_progress(
			opt.progress,
			"${color.build.object}compiling.metax %s",
			sourcefile
		)

		batchcmds:mkdir(
			path.directory(
				objectfile
			)
		)

		batchcmds:vrunv(
			mxcc,
			{
				"-std=c++17",

				"-x",
				"maca",

				"-offload-arch",
				"native",

				"--maca-path="
					.. maca_path,

				"-DENABLE_METAX_API",

				"-fPIC",

				"-Iinclude",
				"-Isrc",

				"-I"
					.. path.join(
						maca_path,
						"include"
					),

				"-c",
				sourcefile,

				"-o",
				objectfile,
			}
		)

		batchcmds:add_depfiles(
			sourcefile
		)

		batchcmds:set_depmtime(
			os.mtime(
				objectfile
			)
		)

		batchcmds:set_depcache(
			target:dependfile(
				objectfile
			)
		)
	end)
rule_end()



-- ============================================================
-- MetaX Operators
-- ============================================================

target("llaisys-ops-metax")
	set_kind("static")

	add_rules(
		"metax.kernel"
	)

	add_files(
		"../src/ops/add/metax/*.maca"
	)

	on_install(function (target)
	end)
target_end()