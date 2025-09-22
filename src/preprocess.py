import jieba
import re
import time

# ---------------------- 加载论文领域自定义词典（核心优化） ----------------------
# 词典文件需与preprocess.py同目录，内容格式："术语 词频 词性"（词频和词性可选）
jieba.load_userdict("paper_dict.txt")


# ---------------------- 加载停用词集合 ----------------------
def load_stopwords():
    """加载论文场景停用词（内置常用停用词，无需外部文件）"""
    return {
        '的' , '是' , '在' , '了' , '我' , '你' , '他' , '她' , '它' , '们' , '和' , '或' , '而' , '就' , '都' ,
        '这' , '那' , '个' , '件' , '条' , '只' , '为' , '以' , '于' , '上' , '下' , '左' , '右' , '前' , '后' ,
        '也' , '还' , '再' , '又' , '不' , '没' , '有' , '着' , '过' , '呢' , '吗' , '吧' , '啊' , '摘要' , '关键词' ,
        '引言' , '正文' , '结论' , '参考文献' , '致谢' , '第一章' , '第二章' , '第一节' , '第二节' , '第三节'
    }


# 全局停用词集合（程序启动时加载一次）
STOPWORDS = load_stopwords()


# ---------------------- 文本清洗 ----------------------
def clean_text(text):
    """去除标点、特殊字符和多余空格，返回清洗后的文本"""
    if not isinstance(text , str):
        return ""
    # 保留中文字符、字母、数字和空格，去除其他所有字符
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]' , '' , text)
    # 合并连续空格为单个空格
    cleaned = re.sub(r'\s+' , ' ' , cleaned).strip()
    return cleaned


# ---------------------- 分词与停用词过滤（含计时逻辑） ----------------------
def segment_text(text):
    """
    对清洗后的文本进行分词和停用词过滤，返回处理后的词列表
    同时计算分词和过滤步骤耗时，打印占比分析
    """
    if not text:
        return []

    # 1. 分词步骤计时（核心优化：使用自定义词典减少歧义）
    start_cut = time.time()
    words = jieba.lcut(text , cut_all = False)  # 精确模式分词（默认）
    cut_time = time.time() - start_cut  # 分词耗时（秒）

    # 2. 停用词过滤步骤计时（基于集合查找，效率O(1)）
    start_filter = time.time()
    filtered_words = [
        word for word in words
        if word.strip() and word not in STOPWORDS and len(word) > 1  # 过滤空字符串、停用词和单字
    ]
    filter_time = time.time() - start_filter  # 过滤耗时（秒）

    # 3. 打印耗时占比分析（保留4位小数，便于优化效果验证）
    total_time = cut_time + filter_time
    print(f"【segment_text】总耗时: {total_time:.4f}s | "
          f"分词耗时: {cut_time:.4f}s ({cut_time / total_time * 100:.2f}%) | "
          f"过滤耗时: {filter_time:.4f}s ({filter_time / total_time * 100:.2f}%)")

    return filtered_words


# ---------------------- 文本预处理主函数 ----------------------
def preprocess(text):
    """整合文本清洗和分词，返回预处理后的词列表字符串（空格分隔）"""
    cleaned_text = clean_text(text)
    tokens = segment_text(cleaned_text)
    return ' '.join(tokens)  # 输出格式适配TF-IDF向量化要求