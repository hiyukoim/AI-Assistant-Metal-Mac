from __future__ import annotations

import asyncio
import atexit
import os
import socket
import sys
import time
import webbrowser
from threading import Event

from fastapi.responses import RedirectResponse

from AI_Assistant_modules.application_config import ApplicationConfig
from AI_Assistant_modules.tab_gui import gradio_tab_gui
from utils.lang_util import LangUtil, get_language_argument

# Apple Silicon Mac port: replace Forge with a ComfyUI sidecar fronted by
# a small a1111-compatible HTTP shim. All Forge imports are gated behind
# IS_MAC because the Forge `modules/` and `modules_forge/` directories
# do not exist in the Mac install. The Windows code path below is
# untouched when IS_MAC is False.
IS_MAC = sys.platform == "darwin"

if IS_MAC:
    from mac.comfy_shim import register_shim

    class _MacStartupTimer:
        """Lightweight stand-in for Forge's modules.timer.startup_timer.
        Records nothing, returns wall-clock since module load on summary().
        """
        def __init__(self):
            self._t0 = time.time()

        def record(self, label):  # noqa: D401 — match Forge's signature
            return

        def summary(self):
            return f"{time.time() - self._t0:.1f}s"

    startup_timer = _MacStartupTimer()
else:
    from modules import initialize
    from modules import initialize_util
    from modules import timer
    from modules_forge import main_thread
    from modules_forge.initialization import initialize_forge

    startup_timer = timer.startup_timer

startup_timer.record("launcher")

if not IS_MAC:
    initialize_forge()
    initialize.imports()
    initialize.check_versions()
    initialize.initialize()

shutdown_event = Event()

def create_api(app):
    # Forge-only — Mac path uses register_shim() instead.
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(starting_port):
    port = starting_port
    while is_port_in_use(port):
        print(f"Port {port} is in use, trying next one.")
        port += 1
    return port

async def api_only_worker(shutdown_event: Event):
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI()

    # 言語設定の取得
    lang_util = LangUtil(get_language_argument())
    # 基準ディレクトリの取得
    if getattr(sys, 'frozen', False):
        # PyInstaller でビルドされた場合
        dpath = os.path.dirname(sys.executable)
    else:
        # 通常の Python スクリプトとして実行された場合
        dpath = os.path.dirname(sys.argv[0])
    app_config = ApplicationConfig(lang_util, dpath)

    #sys.argvの中に--exuiがある場合、app_configにexuiを設定する
    if "--exui" in sys.argv:
        app_config.exui = True

    is_share = False
    if "--share" in sys.argv:
        is_share = True

    # Gradioインターフェースの設定
    _, gradio_url, _ = gradio_tab_gui(app_config).queue().launch(share=is_share, prevent_thread_lock=True)

    # FastAPIのルートにGradioのURLへのリダイレクトを設定
    @app.get("/", response_class=RedirectResponse)
    async def read_root():
        return RedirectResponse(url=gradio_url)

    if IS_MAC:
        # Mac path: mount the a1111-compatible shim instead of Forge's API.
        # The shim translates /sdapi/v1/* calls into ComfyUI workflows and
        # proxies them to the sidecar at 127.0.0.1:8188. For #v4 this is a
        # no-op stub; #v5 wires up trivial endpoints; #v6 lands img2img.
        register_shim(app)
    else:
        initialize_util.setup_middleware(app)
        api = create_api(app)

        from modules import script_callbacks
        script_callbacks.before_ui_callback()
        script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    print(f"Web UI is running at {gradio_url}.")
    webbrowser.open(gradio_url)

    starting_port = 7861
    port = find_available_port(starting_port)
    app_config.set_fastapi_url(f"http://127.0.0.1:{port}")

    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config=config)

    loop = asyncio.get_event_loop()
    shutdown_event.set()
    await loop.create_task(server.serve())

def api_only():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(api_only_worker(shutdown_event))

def on_exit():
    print("Cleaning up...")
    shutdown_event.set()

if __name__ == "__main__":
    atexit.register(on_exit)
    api_only()
    if not IS_MAC:
        # Forge's UI worker thread keeps the process alive after api_only()
        # returns. On Mac, uvicorn's server.serve() inside api_only() is
        # already blocking, so there's nothing else to wait on.
        main_thread.loop()
