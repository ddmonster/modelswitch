from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

logger = logging.getLogger(__name__)


class ConfigWatcher:
    """监听 config.yaml 文件变更，防抖后热重载"""

    def __init__(self, config_path: str, reload_callback):
        self.config_path = str(Path(config_path).resolve())
        self.reload_callback = reload_callback
        self._observer = Observer()
        self._debounce_timer = None
        self._debounce_seconds = 2
        self._loop = None
        self._started = False

    def start(self) -> None:
        """启动文件监听（非阻塞，在新线程中运行）"""
        config_dir = str(Path(self.config_path).parent)
        self._loop = asyncio.new_event_loop()

        handler = _ConfigFileHandler(self.config_path, self._on_file_changed)
        self._observer.schedule(handler, config_dir)
        self._observer.daemon = True
        self._observer.start()
        self._started = True
        logger.info(f"配置热重载已启动，监听 {self.config_path}")

    def stop(self) -> None:
        if self._started:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._started = False
            logger.info("配置热重载已停止")

    def _on_file_changed(self) -> None:
        """防抖：多次快速保存只触发一次重载"""
        if self._debounce_timer:
            self._debounce_timer.cancel()
        self._debounce_timer = threading.Timer(self._debounce_seconds, self._do_reload)
        self._debounce_timer.daemon = True
        self._debounce_timer.start()

    def _do_reload(self) -> None:
        try:
            logger.info("检测到配置变更，开始热重载...")
            self.reload_callback()
            logger.info("配置热重载成功")
        except Exception as e:
            logger.error(f"配置热重载失败: {e}，继续使用旧配置")


class _ConfigFileHandler(FileSystemEventHandler):
    """watchdog 文件事件处理器"""

    def __init__(self, config_path: str, callback):
        self.config_path = config_path
        self.callback = callback

    def on_modified(self, event):
        if event.is_directory:
            return
        if hasattr(event, "src_path") and event.src_path == self.config_path:
            self.callback()
