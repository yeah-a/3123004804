import re
import jieba

# 加载论文领域自定义词典（确保paper_dict.txt存在于项目根目录）
try:
    jieba.load_userdict('paper_dict.txt')
except FileNotFoundError:
    print("Warning: paper_dict.txt not found, using default dictionary.")


def load_stopwords():
    default_stopwords = {
                        '的', '是', '在', '了', '我', '你', '他', '她', '它', '们', '和', '或', '而', '就', '都',
                        '这', '那', '个', '件', '条', '只', '为', '以', '于', '上', '下', '左', '右', '前', '后',
                        '也', '还', '再', '又', '不', '没', '有', '着', '过', '呢', '吗', '吧', '啊', '摘要', '关键词',
                        '引言', '正文', '结论', '参考文献', '致谢', '第一章', '第二章', '第一节', '第二节'
    }
    try:
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        print("Warning: stopwords.txt not found, using default stopwords set.")
        return default_stopwords


STOPWORDS = load_stopwords()


def clean_text(text):
    """
    文本清洗：去除标点/特殊字符，中文与英文/数字间添加空格，合并连续空格
    """
    if not text:
        return ""
    # 1. 移除所有非中文、英文、数字、空格的字符（排除下划线）
    cleaned = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fa5]', '', text)
    # 2. 中文与英文/数字之间添加空格（解决中英文拼接问题）
    cleaned = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z0-9])', r'\1 \2', cleaned)  # 中文后接英文/数字
    cleaned = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fa5])', r'\1 \2', cleaned)  # 英文/数字后接中文
    # 3. 合并连续空格并去除首尾空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def segment_text(text):
    """
    文本分词：使用jieba精确模式，过滤停用词和空字符串
    """
    if not text:
        return []
    words = jieba.lcut(text, cut_all=False)  # 精确模式分词
    filtered_words = [word for word in words if word.strip() and word not in STOPWORDS]
    return filtered_words


def preprocess(text):
    """
    预处理主函数：清洗→分词→拼接为token字符串
    """
    cleaned_text = clean_text(text)
    tokens = segment_text(cleaned_text)
    return ' '.join(tokens)