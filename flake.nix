{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
          nativeBuildInputs = with pkgs; [];
          buildInputs = with pkgs; [
            cudatoolkit
            linuxPackages.nvidia_x11
          ];
        in
        with pkgs;
        {
          devShells.default = mkShell {
            inherit buildInputs nativeBuildInputs;
            shellHook = ''
                export CUDA_PATH=${pkgs.cudatoolkit}
                export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.linuxPackages.nvidia_x11}/lib
                export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
                export EXTRA_CCFLAGS="-I/usr/include"
            '';
          };
        }
      );
}
