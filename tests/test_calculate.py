import unittest
import re
from src.preprocess import clean_text , segment_text , preprocess , STOPWORDS
from src.similarity import tfidf_vectorize , calculate_similarity


class TestPreprocess(unittest.TestCase):
    def test_clean_text(self):
        test_cases = [
            # 正常场景：含标点、中英文、数字
            ("今天天气真好！Hello, world! 123" , "今天天气真好 Hello world 123") ,
            # 边界场景：空文本、全空格
            ("" , "") ,
            ("   \t\n" , "") ,
            # 异常场景：特殊字符
            ("@#$%^&*()_+{}[]|\\:;\"'<>,.?/~`" , "") ,
            ("中文标点：，。；‘’“”？！——" , "中文标点") ,
        ]
        for input_text , expected in test_cases:
            with self.subTest(input = input_text):
                self.assertEqual(clean_text(input_text) , expected)

    def test_segment_text(self):
        # 临时修改停用词集合，确保测试可控（避免依赖外部stopwords.txt）
        original_stopwords = STOPWORDS.copy()
        STOPWORDS.clear()
        STOPWORDS.update({"的" , "是" , "一种" , "在"})  # 测试用停用词

        test_cases = [
            # 正常场景：含专业术语（依赖自定义词典）
            ("余弦相似度是一种算法" , ["余弦相似度" , "算法"]) ,
            # 边界场景：空文本、全停用词
            ("" , []) ,
            ("的 是 在" , []) ,
            # 异常场景：中英文混合分词
            ("AI技术在NLP领域很重要" , ["AI" , "技术" , "NLP" , "领域" , "很重要"]) ,
        ]
        for input_text , expected in test_cases:
            with self.subTest(input = input_text):
                self.assertEqual(segment_text(input_text) , expected)

        # 恢复原始停用词集合
        STOPWORDS.clear()
        STOPWORDS.update(original_stopwords)

    def test_preprocess(self):
        test_cases = [
            # 正常场景：完整流程
            ("今天的天气是晴朗的！" , "今天 天气 晴朗") ,  # 清洗后分词，过滤停用词“的”“是”
            # 边界场景：空文本
            ("" , "") ,
            # 异常场景：特殊字符+停用词
            ("@#$%这是一个测试案例！" , "这 一个 测试 案例") ,
        ]
        for input_text , expected in test_cases:
            with self.subTest(input = input_text):
                self.assertEqual(preprocess(input_text) , expected)


class TestSimilarity(unittest.TestCase):
    def test_tfidf_vectorize(self):
        texts = [
            "今天天气真好 适合出去玩" ,
            "今天天气不错 适合出去旅游"
        ]
        tfidf_matrix = tfidf_vectorize(texts)
        self.assertEqual(tfidf_matrix.shape , (2 , 8))  # 8个unique token（今天/天气/真好/适合/出去/玩/不错/旅游）

    def test_calculate_similarity(self):
        test_cases = [
            # 正常场景：完全相同文本
            ("今天天气真好" , "今天天气真好" , 1.00) ,
            # 正常场景：部分相似（同义词替换）
            ("今天天气真好" , "今日天气晴朗" , 0.33) ,  # 实际值需根据TF-IDF计算
            # 边界场景：两篇空文本
            ("" , "" , 1.00) ,
            # 边界场景：一篇空文本
            ("今天天气真好" , "" , 0.00) ,
            # 异常场景：超长文本（模拟1GB文件片段，截取前1000字）
            ("测试" * 500 , "测试" * 300 + "无效" * 200 , 0.60) ,  # 预期60%相似
        ]
        for orig , copy , expected in test_cases:
            with self.subTest(orig = orig[:20] , copy = copy[:20]):  # 截断显示用例名
                score = calculate_similarity(orig , copy)
                self.assertAlmostEqual(score , expected , delta = 0.05)  # 允许±0.05误差


if __name__ == "__main__":
    unittest.main()