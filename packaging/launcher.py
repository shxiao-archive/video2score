"""video2score 的启动入口（PyInstaller 打包用）：打开本机网页界面。"""

import sys

from video2score.app.server import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
