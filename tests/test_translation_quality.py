import unittest
from unittest.mock import patch

from src.translation import (
    TranslationQualityError,
    semantic_consistency_issues,
    translation_semantic_warnings,
    translation_quality_issues,
    validate_translation,
)


class TranslationQualityTests(unittest.TestCase):
    def test_chinese_one_classifier_accepts_english_indefinite_article(self):
        self.assertEqual(
            validate_translation("一个图书管理员", "A librarian."),
            "A librarian.",
        )

    def test_valid_english_is_accepted(self):
        self.assertEqual(
            validate_translation("今天介绍研究成果", "Today we present our findings."),
            "Today we present our findings.",
        )

    def test_common_model_wrapper_is_removed(self):
        self.assertEqual(
            validate_translation("你好", "English translation: Hello."),
            "Hello.",
        )

    def test_known_mixed_term_is_replaced_before_validation(self):
        with patch.dict("src.translation._TRANSLATION_GLOSSARY", {"蓝色": "blue"}, clear=True):
            self.assertEqual(validate_translation("蓝色的封面", "A 蓝色 cover."), "A blue cover.")

    def test_unknown_chinese_leak_is_rejected(self):
        with self.assertRaises(TranslationQualityError):
            validate_translation("数据驱动乡村发展", "A data-driven 中国乡村")

    def test_prompt_leak_is_rejected(self):
        issues = translation_quality_issues(
            "你好",
            "<current_segment>Hello</current_segment>",
        )
        self.assertIn("翻译结果包含提示词或标签", issues)

    def test_empty_response_is_rejected(self):
        with self.assertRaises(TranslationQualityError):
            validate_translation("你好", "")

    def test_date_and_time_accept_month_name_and_pm_time(self):
        self.assertEqual(
            semantic_consistency_issues(
                "会议定于2027年4月12日17点开始",
                "The meeting will begin at 5 PM on April 12, 2027.",
            ),
            [],
        )

    def test_changed_year_is_rejected(self):
        issues = semantic_consistency_issues(
            "项目将在二零二六年完成",
            "The project will be completed in 2023.",
        )
        self.assertTrue(any("年份不一致" in issue for issue in issues))

    def test_spelled_out_percentage_is_accepted(self):
        self.assertEqual(
            semantic_consistency_issues(
                "效率提高百分之二十",
                "Efficiency increased by twenty percent.",
            ),
            [],
        )

    def test_changed_currency_amount_is_rejected(self):
        issues = semantic_consistency_issues(
            "项目预算为三百五十万元",
            "The project budget is 3.05 million yuan.",
        )
        self.assertTrue(any("金额不一致" in issue for issue in issues))

    def test_spelled_out_large_currency_amount_is_accepted(self):
        self.assertEqual(
            semantic_consistency_issues(
                "项目预算为三百五十万元",
                "The project budget is three million five hundred thousand yuan.",
            ),
            [],
        )

    def test_decimal_chinese_magnitude_is_checked(self):
        self.assertEqual(
            semantic_consistency_issues(
                "投入约三点五万元",
                "The investment is about 35,000 yuan.",
            ),
            [],
        )
        self.assertTrue(
            semantic_consistency_issues(
                "投入约三点五万元",
                "The investment is about 3,500 yuan.",
            )
        )

    def test_both_contrasted_amounts_must_be_preserved(self):
        issues = semantic_consistency_issues(
            "价格是1000元，不是10000元",
            "The price is 10,000 yuan.",
        )
        self.assertTrue(any("1000元" in issue for issue in issues))

    def test_person_labels_must_be_preserved(self):
        issues = semantic_consistency_issues(
            "A先向B支付10元",
            "The buyer first pays the seller ten yuan.",
        )
        self.assertTrue(any("编号不一致" in issue for issue in issues))

    def test_missing_negation_is_a_retry_warning_not_a_hard_failure(self):
        source = "这并不是最终结果"
        candidate = "This is the final result."
        self.assertEqual(semantic_consistency_issues(source, candidate), [])
        self.assertTrue(translation_semantic_warnings(source, candidate))


if __name__ == "__main__":
    unittest.main()
