add_requires("openmp")

target("llaisys-device-cpu")
	set_kind("static")
	set_languages("cxx17")
	set_warnings("all", "error")

	if not is_plat("windows") then
		add_cxflags("-fPIC", "-Wno-unknown-pragmas")
	end

	add_files("../src/device/cpu/*.cpp")

	on_install(function (target)
	end)
target_end()

target("llaisys-ops-cpu")
	set_kind("static")
	add_deps("llaisys-tensor")
	set_languages("cxx17")
	set_warnings("all", "error")

	if not is_plat("windows") then
		add_cxflags("-fPIC", "-Wno-unknown-pragmas")
	end

	-- Enable OpenMP compilation and propagate its link dependency
	-- to targets that link this static library.
	add_packages("openmp", {public = true})

	add_files("../src/ops/*/cpu/*.cpp")

	on_install(function (target)
	end)
target_end()