"""
ماژول utils برای توابع کمکی
"""
import os
import sys

# اصلاح مسیرهای import
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)