import os

def read_file(path):
    """读取文件内容，处理文件不存在、权限错误、编码错误等异常"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    if not os.path.isfile(path):
        raise IsADirectoryError(f"路径不是文件: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        raise UnicodeDecodeError(f"文件编码错误，请确保使用UTF-8编码: {path}")
    except PermissionError:
        raise PermissionError(f"无权限读取文件: {path}")
    except Exception as e:
        raise RuntimeError(f"读取文件失败: {str(e)}")

def write_result(path, score):
    """写入相似度结果，确保保留两位小数"""
    try:
        # 格式化得分，确保两位小数（如0.8 → 0.80）
        formatted_score = f"{score:.2f}"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(formatted_score)
    except PermissionError:
        raise PermissionError(f"无权限写入文件: {path}")
    except Exception as e:
        raise RuntimeError(f"写入文件失败: {str(e)}")