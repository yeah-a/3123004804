import sys
import traceback
from src.file_utils import read_file , write_result
from src.preprocess import preprocess
from src.similarity import calculate_similarity


def validate_arguments(args):
    """验证命令行参数"""
    if len(args) != 4:
        raise ValueError("参数数量错误！正确用法: python main.py [原文文件] [抄袭版文件] [结果文件]")
    orig_path , copy_path , result_path = args[1] , args[2] , args[3]
    # 简单验证路径格式（避免明显错误）
    if not (orig_path.endswith('.txt') and copy_path.endswith('.txt')):
        raise ValueError("输入文件必须为txt格式")
    return orig_path , copy_path , result_path


def main():
    """主函数：解析参数→读取文件→预处理→计算相似度→写入结果"""
    try:
        # 验证参数
        orig_path , copy_path , result_path = validate_arguments(sys.argv)

        # 读取原文和抄袭版文本
        orig_raw = read_file(orig_path)
        copy_raw = read_file(copy_path)

        # 文本预处理
        orig_processed = preprocess(orig_raw)
        copy_processed = preprocess(copy_raw)

        # 计算相似度
        similarity_score = calculate_similarity(orig_processed , copy_processed)

        # 写入结果（自动保留两位小数）
        write_result(result_path , similarity_score)

    except Exception as e:
        # 捕获所有异常并友好提示
        print(f"程序执行失败: {str(e)}" , file = sys.stderr)
        # 调试时可开启以下行查看详细堆栈信息
        # traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()