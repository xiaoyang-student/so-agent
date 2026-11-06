"""跨平台进程单实例锁。

单用户部署仍可能误开两个 Web 进程：两者会共享 SQLite 并各自启动内置
调度器，导致同一持久化任务被重复执行。本模块用操作系统文件锁阻止
第二个实例启动，不依赖第三方包。

实现差异：
- Windows 使用 ``msvcrt.locking``（锁定文件首字节）；
- POSIX 使用 ``fcntl.flock``（非阻塞独占锁）。

锁文件保留在磁盘上不删除，避免释放与重建之间的竞态；锁本身随进程
退出自动释放。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class InstanceLockError(RuntimeError):
    """单实例锁已被其他进程持有。"""


class InstanceLock:
    """基于文件锁的单实例锁。"""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        """非阻塞获取锁；已被持有时抛 :class:`InstanceLockError`。"""
        if self._fd is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self._lock_fd(fd)
        except OSError as exc:
            os.close(fd)
            raise InstanceLockError(
                f"另一个 so-agent 实例正在使用该目录（锁文件：{self._path}）"
            ) from exc
        self._fd = fd
        self._write_owner(fd)

    def release(self) -> None:
        fd = self._fd
        self._fd = None
        if fd is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _write_owner(self, fd: int) -> None:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(fd)
        except OSError:
            # 锁已持有；写入诊断信息失败不影响互斥语义。
            pass

    @staticmethod
    def _lock_fd(fd: int) -> None:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __enter__(self) -> "InstanceLock":
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
