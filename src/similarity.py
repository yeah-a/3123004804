from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_vectorize(texts):
    """将文本列表转换为TF-IDF向量矩阵"""
    try:
        # 配置TF-IDF参数（保留高频特征，过滤低频词）
        vectorizer = TfidfVectorizer(
            max_features = 10000 ,  # 保留Top 10000高频词
            min_df = 1 ,  # 至少出现在1篇文本中
            ngram_range = (1 , 2)  # 考虑1-gram和2-gram特征
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix
    except ValueError as e:
        if "empty vocabulary" in str(e):
            # 文本预处理后无有效词汇
            return None
        raise RuntimeError(f"TF-IDF向量化失败: {str(e)}")


def calculate_similarity(orig_text , copy_text):
    """计算两篇预处理后文本的余弦相似度"""
    # 处理极端情况
    if not orig_text and not copy_text:
        return 1.0  # 两篇均为空文本，视为完全重复
    if not orig_text or not copy_text:
        return 0.0  # 一篇为空，一篇非空，视为无重复

    # 生成TF-IDF向量
    tfidf_matrix = tfidf_vectorize([orig_text , copy_text])
    if tfidf_matrix is None:
        return 0.0  # 无有效词汇，视为无重复

    # 计算余弦相似度（取值范围[0, 1]）
    similarity_score = cosine_similarity(tfidf_matrix[0:1] , tfidf_matrix[1:2])[0][0]
    return round(similarity_score , 4)  # 保留四位小数用于后续四舍五入