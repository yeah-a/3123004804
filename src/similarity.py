from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def tfidf_vectorize(texts):
    vectorizer = TfidfVectorizer(
        token_pattern=r'\S+',  # 匹配任意非空白字符序列（支持中文分词结果）
        max_features=10000,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix


def calculate_similarity(orig_token_str , copy_token_str):
    # 处理空文本边界情况
    if not orig_token_str and not copy_token_str:
        return 1.0  # 两篇均为空文本
    if not orig_token_str or not copy_token_str:
        return 0.0  # 一篇为空另一篇非空

    # 生成TF-IDF向量（确保输入为列表格式）
    tfidf_matrix = tfidf_vectorize([orig_token_str , copy_token_str])
    # 计算余弦相似度（提取向量并确保维度一致）
    similarity_score = sklearn_cosine(tfidf_matrix[0:1] , tfidf_matrix[1:2])[0][0]
    return round(similarity_score , 4)  # 保留4位小数，避免精度问题