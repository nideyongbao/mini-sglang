"""Reference benchmark server; numerical/continuation hooks are opt-in patches."""

import os
import sys

from audit import install_sglang

install_sglang()

if __name__ == "__main__":
    from sglang.launch_server import kill_process_tree, prepare_server_args, run_server
    from sglang.srt.plugins import load_plugins

    load_plugins()
    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
