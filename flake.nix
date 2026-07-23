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
					llamaCpp = pkgs.fetchFromGitHub {
						owner = "ggml-org";
						repo = "llama.cpp";
						tag = llamaCppVersion;
						hash = "sha256-ZHQ9hBnE9GayZRt0jgO4svzaAUfhRUg6cFu5dSe8J1w=";
					};
					base = if acceleration == "cuda"
						then pkgs.ollama-cuda.override { cudaArches = [ "sm_86" ]; }
						else pkgs.ollama;
				in base.overrideAttrs (finalAttrs: _oldAttrs: {
					pname = "ollama-pmarreck";
					version = "0.32.0-pmarreck.1";
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
