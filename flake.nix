{
	description = "Ollama with opt-in parallel embedding support";

	inputs.nixpkgs.url = "github:NixOS/nixpkgs/753cc8a3a87467296ddd1fa93f0cc3e81120ee46";

	outputs = { self, nixpkgs }:
		let
			systems = [
				"x86_64-linux"
				"aarch64-linux"
				"aarch64-darwin"
			];
			forAllSystems = nixpkgs.lib.genAttrs systems;
			pkgsFor = system: import nixpkgs {
				inherit system;
				config.allowUnfree = true;
			};
			llamaCppVersion = nixpkgs.lib.removeSuffix "\n" (builtins.readFile ./LLAMA_CPP_VERSION);
			mkOllama = { pkgs, acceleration ? null }:
				let
					inherit (pkgs) lib;
					llamaCpp = pkgs.fetchFromGitHub {
						owner = "ggml-org";
						repo = "llama.cpp";
						tag = llamaCppVersion;
						hash = "sha256-HT0QuIFJz5cgH2qinxhtyLEL/RrUpziZuntj/EDQtzI=";
					};
					base = if acceleration == "cuda"
						then pkgs.ollama-cuda.override { cudaArches = [ "sm_86" ]; }
						else pkgs.ollama;

					# A single CUDA root that actually contains bin/nvcc *and* the
					# headers/libs, so toolkit discovery never depends on ambient state.
					# Lazily evaluated: never forced unless acceleration == "cuda".
					cudaLibs = with pkgs.cudaPackages; [ cuda_cudart libcublas cccl ];
					cudaToolkitRoot = pkgs.buildEnv {
						name = "ollama-cuda-toolkit-root";
						paths = [
							(lib.getBin pkgs.cudaPackages.cuda_nvcc)
							(lib.getDev pkgs.cudaPackages.cuda_nvcc)
							(lib.getOutput "static" pkgs.cudaPackages.cuda_cudart)
						]
						++ map lib.getLib cudaLibs
						++ map lib.getDev cudaLibs;
						# cccl and cuda_cudart both ship a LICENSE at the same path.
						ignoreCollisions = true;
					};
				in base.overrideAttrs (finalAttrs: oldAttrs: {
					pname = "ollama-pmarreck";
					version = "0.32.13-pmarreck.1";
					src = self;

					postPatch = ''
						substituteInPlace version/version.go \
							--replace-fail 0.0.0 '${finalAttrs.version}'

						# These launcher tests install third-party programs through npm.
						rm cmd/launch/*_test.go
						rm -r app

						# CMake's FetchContent cannot access the network in the Nix sandbox.
						cp -r ${llamaCpp} $TMPDIR/llama-cpp-src
						chmod -R +w $TMPDIR/llama-cpp-src
						( cd $TMPDIR/llama-cpp-src && \
							cmake -DPATCH_DIR=$NIX_BUILD_TOP/source/llama/compat \
								-P $NIX_BUILD_TOP/source/llama/compat/apply-patch.cmake )
					'';
				} // lib.optionalAttrs (acceleration == "cuda") {
					# Pin CUDA toolkit discovery to one root that provably contains
					# bin/nvcc, instead of letting three ambient channels race.
					#
					# WHY: CMake's FindCUDAToolkit changed between 4.1 and 4.3. Once
					# CUDAToolkit_ROOT is set, 4.3+ searches ONLY there (NO_DEFAULT_PATH)
					# and no longer falls back to PATH; 4.1 does fall back. Nixpkgs'
					# setupCudaHook builds CUDAToolkit_ROOT from cudaHostPathsSeen, which
					# is populated from host/target deps only -- and cuda_nvcc is a *native*
					# build input, so it can be absent from the very list CMake is told to
					# search for nvcc. On cmake >= 4.3 that combination is a hard failure:
					#   -- Could not find `nvcc` executable in path specified by
					#      environment variable CUDAToolkit_ROOT=...
					#   CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:268 (message):
					#     CUDA Toolkit not found
					# Verified by differential probe: identical CUDA packages and an
					# identical nvcc-less CUDAToolkit_ROOT, nvcc on PATH in both --
					# cmake 4.1.2 => FOUND, cmake 4.3.4 => NOT FOUND. This build only
					# works today because nixpkgs currently ships cmake 4.1.2; it would
					# break again on the next bump past 4.3.
					#
					# Also overrides CUDA_PATH, which nixpkgs sets to
					#   lib.removeSuffix "-${cudaMajorVersion}" cudaToolkit
					# -- stripping "-12" yields a path that does NOT exist. A fallback
					# that can never fire also can never be noticed as broken.
					#
					# BOLD ALTERNATIVE (not done here, deliberately): the fully correct
					# fix is to pass -DCUDAToolkit_ROOT as a CMake *cache* variable rather
					# than an env var, because ollama's cmake/local.cmake forwards toolkit
					# args to the cuda_v12 ExternalProject child via
					# ollama_append_cache_arg_if_set(), which only forwards cache entries.
					# That requires replacing nixpkgs' preBuild wholesale (duplicating its
					# cudaArchitectures / llamaBackend / preset logic), because ollama
					# hand-rolls `cmake -B build ...` in preBuild and references neither
					# $cmakeFlags nor "${cmakeFlagsArray[@]}" -- so every -D the CUDA setup
					# hook computes (CUDAToolkit_ROOT, CUDAToolkit_INCLUDE_DIR,
					# CMAKE_CUDA_HOST_COMPILER) is silently dropped. The upstream-correct
					# version of this fix is a nixpkgs PR that (a) drops the removeSuffix,
					# (b) passes -DCUDAToolkit_ROOT=${cudaToolkit}, and (c) consumes
					# cmakeFlagsArray in preBuild. Env-var pinning below is chosen instead
					# because it is version-proof under both CMake semantics, needs no
					# duplication of upstream logic, and reverts in one line.
					preBuild = ''
						export CUDAToolkit_ROOT=${cudaToolkitRoot}
						export CUDA_PATH=${cudaToolkitRoot}
						if [ ! -x "${cudaToolkitRoot}/bin/nvcc" ]; then
							echo "CUDA toolkit root lacks bin/nvcc: ${cudaToolkitRoot}" >&2
							exit 1
						fi
					'' + oldAttrs.preBuild;
				});
		in {
			packages = forAllSystems (system:
				let
					pkgs = pkgsFor system;
					cpu = mkOllama { inherit pkgs; };
				in {
					inherit cpu;
					default = if system == "x86_64-linux"
						then mkOllama { inherit pkgs; acceleration = "cuda"; }
						else cpu;
				} // nixpkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
					test-env = pkgs.buildFHSEnv {
						name = "ollama-test-env";
						targetPkgs = testPkgs: with testPkgs; [
							bash
							coreutils
							findutils
							gawk
							gitMinimal
							go
							gnugrep
							gnused
						];
						runScript = "bash";
					};
				});

			checks = forAllSystems (system: {
				build = self.packages.${system}.default;
			});

			devShells = forAllSystems (system:
				let pkgs = pkgsFor system;
				in {
					default = pkgs.mkShell {
						packages = with pkgs; [
							cmake
							curl
							gitMinimal
							go
							jq
							ninja
							pkg-config
							shellcheck
						];
					};
				});
		};
}
