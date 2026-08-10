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