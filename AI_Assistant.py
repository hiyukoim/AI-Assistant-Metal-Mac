import locale
import sys

# Apple Silicon Mac port: this file's only job is to (1) drop --xformers
# from default_args (no Apple Silicon build of xformers exists), and
# (2) bypass modules.launch_utils_AI_Assistant on Mac, which imports
# Forge packages we don't ship. The Windows code path below is unchanged
# when IS_MAC is False — same import, same args, same start function.
IS_MAC = sys.platform == "darwin"

# Default arguments
if IS_MAC:
    default_args = ["--nowebui", "--skip-python-version-check", "--skip-torch-cuda-test", "--skip-torch-cuda-test"]
else:
    default_args = ["--nowebui", "--xformers", "--skip-python-version-check", "--skip-torch-cuda-test", "--skip-torch-cuda-test"]

# Check if custom arguments are provided; if not, append default arguments
if len(sys.argv) == 1:
    sys.argv.extend(default_args)
else:
    # 独自の引数がある場合、default_argsの中で未指定の引数のみを追加する
    # 引数を解析しやすくするため、setを使用
    provided_args_set = set(sys.argv)
    for arg in default_args:
        # "--"で始まるオプションのみを考慮する
        if arg.startswith("--"):
            option = arg.split("=")[0] if "=" in arg else arg
            if option not in provided_args_set and "--no-" + option.removeprefix('--') not in provided_args_set:
                sys.argv.append(arg)
        else:
            # "--"で始まらないオプションは直接追加
            sys.argv.append(arg)

if "--lang" not in sys.argv:
    system_locale = locale.getdefaultlocale()[0]
    if system_locale.startswith("ja"):
        sys.argv.append("--lang=jp")
    elif system_locale.startswith("zh"):
        sys.argv.append("--lang=zh_CN")
    else:
        sys.argv.append("--lang=en")

if IS_MAC:
    # On Mac, modules.launch_utils_AI_Assistant cannot be imported because
    # the Forge `modules/` directory does not exist. Replicate the minimum
    # behaviour inline: print the launch banner and call AI_Assistant_gui.
    args = None  # consumers on the Mac path do not use this
    def start():
        print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {' '.join(sys.argv[1:])}")
        import AI_Assistant_gui
        AI_Assistant_gui.api_only()
else:
    from modules import launch_utils_AI_Assistant
    args = launch_utils_AI_Assistant.args
    start = launch_utils_AI_Assistant.start

def main():
    start()

if __name__ == "__main__":
    main()
