{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # nixpkgs.url = "github:NixOS/nixpkgs/efcb904a6c674d1d3717b06b89b54d65104d4ea7";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        libList = [
          pkgs.stdenv.cc.cc
          pkgs.stdenv.cc
          pkgs.libGL
          pkgs.gcc
          pkgs.glib
          pkgs.libz
          pkgs.glibc
        ];
      in
      with pkgs;
      rec {
        scripts = let
            python_with_pkgs = python3.withPackages (pp: [
              (inputs.nahual-flake.packages.${system}.nahual)
              packages.vit
            ]);
            in 
              {
                runMorphem = pkgs.writeScriptBin "run_morphem" ''
                 #!${pkgs.bash}/bin/bash
                    ${python_with_pkgs}/bin/python ${self}/src/vit/morphem.py ''${1:-"ipc:///tmp/morphem.ipc"}
                      '';
                runOpenphenom = pkgs.writeScriptBin "run_openphenom" ''
                  #!${pkgs.bash}/bin/bash
                   ${python_with_pkgs}/bin/python ${self}/src/vit/openphenom.py ''${1:-"ipc:///tmp/openphenom.ipc"}
                '';
              };
        apps = rec {
            morphem = {
              type = "app";
              program = "${self.scripts.${stdenv.hostPlatform.system}.runMorphem}/bin/run_morphem";
            };
            openphenom = {
              type = "app";
              program = "${self.scripts.${stdenv.hostPlatform.system}.runOpenphenom}/bin/run_openphenom";
            };
            default = morphem;
          };

        packages = {
          vit = pkgs.python3.pkgs.callPackage ./nix/vit.nix { };
        };
        devShells = {
          default =
            let
              python_with_pkgs = (
                python3.withPackages (pp: [
                  (inputs.nahual-flake.packages.${system}.nahual)
                  packages.vit
                ])
              );
            in
            mkShell {
              packages = [
                python_with_pkgs
                python3Packages.venvShellHook
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
              ];
              currentSystem = system;
              venvDir = "./.venv";
              postVenvCreation = ''
                unset SOURCE_DATE_EPOCH
              '';
              postShellHook = ''
                unset SOURCE_DATE_EPOCH
              '';
              shellHook = ''
                # Set PYTHONPATH to only include the Nix packages, excluding current directory
                runHook venvShellHook
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}
              '';
            };

          # Minimal shell for the pixi-based path. Exposes pixi and the system
          # NVIDIA driver libs so conda-installed pytorch-gpu can load libcuda
          # on NixOS, where /run/opengl-driver/lib is not on a default search
          # path. Use as:  nix develop .#pixi --command pixi run morphem ...
          # (Pattern from shntnu/neusis templates/python-pixi.)
          pixi = mkShell {
            packages = [ pkgs.pixi ];
            LD_LIBRARY_PATH = "/run/opengl-driver/lib";
          };
        };
      }
    );
}
# export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
# export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
# export NVCC_APPEND_FLAGS="-Xcompiler -fno-PIC"
# export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
# export CUDA_NVCC_FLAGS="-O2 -Xcompiler -fno-PIC"
# # Ensure current directory is not in Python path
# export PYTHONDONTWRITEBYTECODE=1
