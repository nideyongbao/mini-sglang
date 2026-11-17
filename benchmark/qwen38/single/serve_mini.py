"""Benchmark-only server: forced continuation, including spawned scheduler."""

from audit import install_mini

install_mini()

if __name__ == "__main__":
    from minisgl.server import launch_server

    launch_server()
