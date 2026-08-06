-- ============================================================
-- NVIDIA Runtime
-- ============================================================

target("llaisys-device-nvidia")
set_kind("static")
set_languages("cxx17")
set_warnings("all", "error")

set_policy("build.cuda.devlink", true)
set_toolchains("cuda")

add_links("cudart")

if is_plat("windows") then
    add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")

    add_cxxflags("/FS")
else
    add_cuflags("-Xcompiler=-fPIC")
    add_culdflags("-Xcompiler=-fPIC")
    add_cxxflags("-fPIC")
end

add_files("../src/device/nvidia/*.cu")

on_install(
    function(target)
    end
)

    on_config(function (target)
        local cuda_root = get_config("cuda")

        if not cuda_root then
            raise(
                "CUDA Toolkit directory was not detected"
            )
        end

        local cuda_include_dir = path.join(
            cuda_root,
            "include"
        )

        target:add(
            "includedirs",
            cuda_include_dir,
            {public = true}
        )

        cprint(
            "${cyan}NVIDIA device headers: %s",
            cuda_include_dir
        )
    end)
    
target_end()

-- ============================================================
-- NVIDIA Operators
-- ============================================================

target("llaisys-ops-nvidia")
set_kind("static")

add_deps("llaisys-tensor")

set_languages("cxx17")
set_warnings("all", "error")

set_policy("build.cuda.devlink", true)
set_toolchains("cuda")

add_links("cudart")
add_links("cublas")

add_cugencodes("native")

if is_plat("windows") then
    add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")

    add_cuflags("-Xcompiler=/W3", "-Xcompiler=/WX")

    add_cxxflags("/FS")
else
    add_cuflags("-Xcompiler=-Wall", "-Xcompiler=-Werror")

    add_cuflags("-Xcompiler=-fPIC")
    add_cuflags("--extended-lambda")
    add_cuflags("--expt-relaxed-constexpr")

    add_cuflags("-Xcompiler=-Wno-error=deprecated-declarations")

    add_culdflags("-Xcompiler=-fPIC")
    add_cxxflags("-fPIC")
end

add_files("../src/ops/*/nvidia/*.cu")

-- ========================================================
-- Optional system cuDNN Frontend detection
-- ========================================================

on_config(
    function(target)
        import("lib.detect.find_file")
        import("lib.detect.find_library")

        if not is_plat("linux") then
            cprint("${yellow}Automatic cuDNN detection is currently enabled only on Linux")

            cprint("${yellow}Using CUDA self-attention fallback")

            return
        end

        -- Xmake-detected CUDA Toolkit root.
        -- On this machine it resolves to /usr/local/cuda,
        -- but that path is not hardcoded here.
        local cuda_root = get_config("cuda")

        if not cuda_root or not os.isdir(cuda_root) then
            cprint("${yellow}CUDA Toolkit root was not detected")

            cprint("${yellow}Using CUDA self-attention fallback")

            return
        end

        -- ----------------------------------------------------
        -- Architecture-specific system paths
        -- ----------------------------------------------------

        local arch = os.arch()
        local linux_multiarch = nil
        local cuda_target_arch = nil

        if arch == "x86_64" then
            linux_multiarch = "x86_64-linux-gnu"
            cuda_target_arch = "x86_64-linux"
        elseif arch == "arm64" or arch == "aarch64" then
            linux_multiarch = "aarch64-linux-gnu"
            cuda_target_arch = "aarch64-linux"
        end

        -- ----------------------------------------------------
        -- Search roots and suffixes
        -- ----------------------------------------------------

        local system_roots = {
            "/usr",
            "/usr/local"
        }

        local system_include_suffixes = {
            "/include"
        }

        local system_library_suffixes = {
            "/lib",
            "/lib64"
        }

        if linux_multiarch then
            table.insert(system_include_suffixes, "/include/" .. linux_multiarch)

            table.insert(system_library_suffixes, "/lib/" .. linux_multiarch)

            table.insert(system_library_suffixes, "/lib64/" .. linux_multiarch)
        end

        local cuda_include_suffixes = {
            "/include"
        }

        local cuda_library_suffixes = {
            "/lib64",
            "/lib"
        }

        if cuda_target_arch then
            table.insert(cuda_include_suffixes, "/targets/" .. cuda_target_arch .. "/include")

            table.insert(cuda_library_suffixes, "/targets/" .. cuda_target_arch .. "/lib")
        end

        -- Do not add CUDA stub directories. The stub libraries
        -- are not valid runtime implementations.

        -- ----------------------------------------------------
        -- Detect headers
        -- ----------------------------------------------------

        local cudnn_backend_header =
            find_file(
            "cudnn.h",
            system_roots,
            {
                suffixes = system_include_suffixes
            }
        )

        local cudnn_frontend_header =
            find_file(
            "cudnn_frontend.h",
            system_roots,
            {
                suffixes = system_include_suffixes
            }
        )

        local nvrtc_header =
            find_file(
            "nvrtc.h",
            {
                cuda_root
            },
            {
                suffixes = cuda_include_suffixes
            }
        )

        -- ----------------------------------------------------
        -- Detect shared libraries
        -- ----------------------------------------------------

        local cudnn_library =
            find_library(
            "cudnn",
            system_roots,
            {
                kind = "shared",
                suffixes = system_library_suffixes
            }
        )

        local nvrtc_library =
            find_library(
            "nvrtc",
            {
                cuda_root
            },
            {
                kind = "shared",
                suffixes = cuda_library_suffixes
            }
        )

        -- ----------------------------------------------------
        -- Report detection results
        -- ----------------------------------------------------

        if cudnn_backend_header then
            cprint("${green}cuDNN Backend header: %s", cudnn_backend_header)
        else
            cprint("${yellow}System cudnn.h not found")
        end

        if cudnn_frontend_header then
            cprint("${green}cuDNN Frontend header: %s", cudnn_frontend_header)
        else
            cprint("${yellow}System cudnn_frontend.h not found")
        end

        if nvrtc_header then
            cprint("${green}NVRTC header: %s", nvrtc_header)
        else
            cprint("${yellow}nvrtc.h not found")
        end

        if cudnn_library then
            cprint("${green}cuDNN library: %s", cudnn_library.filename)
        else
            cprint("${yellow}System libcudnn shared library not found")
        end

        if nvrtc_library then
            cprint("${green}NVRTC library: %s", nvrtc_library.filename)
        else
            cprint("${yellow}libnvrtc shared library not found")
        end

        -- ----------------------------------------------------
        -- Enable cuDNN only when all dependencies are present
        -- ----------------------------------------------------

        if cudnn_backend_header and cudnn_frontend_header and nvrtc_header and cudnn_library and nvrtc_library then
            -- CUDA 12.8 include directory must come before the
            -- old Ubuntu CUDA 11.5 headers in /usr/include.
            target:add("includedirs", path.directory(nvrtc_header), {public = true})

            target:add("includedirs", path.directory(cudnn_backend_header), {public = true})

            target:add("includedirs", path.directory(cudnn_frontend_header), {public = true})

            target:add("linkdirs", nvrtc_library.linkdir, {public = true})

            target:add("linkdirs", cudnn_library.linkdir, {public = true})

            target:add("defines", "ENABLE_CUDNN_API")

            -- These are system libraries required by the
            -- objects inside this static target. Public
            -- visibility propagates them to libllaisys.so.
            target:add("syslinks", "cudnn", {public = true})

            target:add("syslinks", "nvrtc", {public = true})

            -- Embed the detected runtime search paths so that
            -- ctypes can load libllaisys.so without requiring
            -- a manually configured LD_LIBRARY_PATH.
            target:add("rpathdirs", nvrtc_library.linkdir, {public = true})

            target:add("rpathdirs", cudnn_library.linkdir, {public = true})

            cprint("${green}cuDNN Self-Attention enabled")
        else
            cprint("${yellow}Using CUDA self-attention fallback")
        end
    end
)

on_install(
    function(target)
    end
)
target_end()
