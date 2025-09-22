import re
import jieba

def load_stopwords():
    """加载停用词表，优先从本地文件读取，不存在则使用默认列表"""
    stopwords = set()
    # 尝试加载本地停用词文件
    try:
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set(f.read().splitlines())
    except FileNotFoundError:
        # 默认停用词列表（适用于论文场景）
        stopwords = {
            '的', '是', '在', '了', '我', '你', '他', '她', '它', '们', '和', '或', '而', '就', '都',
            '这', '那', '个', '件', '条', '只', '为', '以', '于', '上', '下', '左', '右', '前', '后',
            '也', '还', '再', '又', '不', '没', '有', '着', '过', '呢', '吗', '吧', '啊', '摘要', '关键词',
            '引言', '正文', '结论', '参考文献', '致谢', '第一章', '第二章', '第三章', '第一节', '第二节'
        }
    return stopwords

# 全局停用词集合（程序启动时加载一次）
STOPWORDS = load_stopwords()

def clean_text(text):
    """清洗文本：去除标点、特殊字符和多余空格"""
    if not isinstance(text, str):
        return ""
    # 保留中文字符、字母、数字和空格，去除其他所有字符
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    # 合并连续空格为单个空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# preprocess.py（修改后）
import time  # 导入时间模块
import jieba
from multiprocessing import cpu_count


# 分词并过滤停用词，返回处理后的词列表
def segment_text(text):
    if not text:
        return []
    start_cut = time.time()  # 分词开始时间
    words = jieba.lcut(text , cut_all = False)  # 精确模式分词
    cut_time = time.time() - start_cut  # 分词耗时（秒）

    start_filter = time.time()  # 过滤开始时间
    filtered_words = [word for word in words if word.strip() and word not in STOPWORDS]  # 过滤逻辑
    filter_time = time.time() - start_filter  # 过滤耗时（秒）

    total_time = cut_time + filter_time
    print(
        f"【segment_text】总耗时: {total_time:.4f}s | 分词耗时: {cut_time:.4f}s ({cut_time / total_time * 100:.2f}%) "
        f"| 过滤耗时: {filter_time:.4f}s ({filter_time / total_time * 100:.2f}%)")

    return filtered_words


# 文本预处理主函数：清洗→分词→拼接为空格分隔的字符串
def preprocess(text):

    cleaned_text = clean_text(text)
    tokens = segment_text(cleaned_text)
    return ' '.join(tokens)  # 适合TF-IDF向量化的输入格式