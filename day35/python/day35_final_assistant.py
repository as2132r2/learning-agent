"""毕业项目固定入口。

python day35/python/day35_final_assistant.py xiaolin-capstone-001
python day35/python/day35_final_assistant.py --demo
python day35/python/day35_final_assistant.py --check

业务实现复用Day34，保持原有数据目录与命令行为。
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "day34/python"))

from day34_learning_assistant import main  # noqa: E402


if __name__ == "__main__":
    main()
