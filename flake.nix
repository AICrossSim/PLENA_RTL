{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    systems.url = "github:nix-systems/default-linux";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs: let
    lib = nixpkgs.lib;
  in
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
      llvm14 = pkgs.llvmPackages_14;
    in rec {
      # ---------- Formatter ----------
      formatter = pkgs.alejandra;

      # ---------- Development Shells ----------
      devShells = {
        default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # --- Verilog/SystemVerilog toolchain ---
            verilator
            verible

            # --- Compilers / build tools ---
            gcc
            gnumake
            cmake
            ninja
            pkg-config
            autoconf
            flex
            bison
            ccache
            help2man

            # --- LLVM/Clang (plus specific 14.x, if needed) ---
            clang
            llvmPackages.clang-unwrapped
            llvmPackages.lld
            clang-tools
            llvm14.clang
            llvm14.lld

            # --- General dev / utils ---
            git
            wget
            unzip
            vim
            htop
            xdg-utils
            parallel
            just

            # --- Crypto / SSL / IDN ---
            openssl
            libidn

            # --- Python ---
            python312
            python312Packages.pip

            # --- Graphics / docs ---
            graphviz
          ];

          nativeBuildInputs = with pkgs; [
            uv
          ];

          shellHook = ''
            export PYTHONPATH="$PWD:$PWD/tools:''${PYTHONPATH:-}"

            echo ">>> Toolchain versions:"
            echo "Verilator:    $(verilator --version 2>/dev/null || echo not found)"
            echo "Verible:      $(verible-verilog-format --version 2>/dev/null || echo not found)"
            echo "Clang:        $(clang --version | head -n1 2>/dev/null || echo not found)"
            echo "GCC:          $(gcc --version | head -n1 2>/dev/null || echo not found)"
            echo "CMake:        $(cmake --version | head -n1 2>/dev/null || echo not found)"
            echo "Python 3.12:  $(python3.12 --version 2>/dev/null || echo not found)"
          '';
        };
      };
    });
}
