from shared.src.shared.catalog import SUPPORTED_PAIRS, describe_supported, is_supported


def test_chat_supports_gre_quant_and_aw_and_the_ielts_coaches():
    assert set(SUPPORTED_PAIRS) == {
        ("gre", "quant"), ("gre", "aw"), ("ielts", "mock"), ("ielts", "reading"),
        ("ielts", "writing"),
    }
    assert is_supported("gre", "quant") and is_supported("gre", "aw")
    assert is_supported("ielts", "writing") and is_supported("ielts", "reading")
    assert is_supported("ielts", "mock")
    assert not is_supported("gre", "verbal")
    assert not is_supported("ielts", "speaking")


def test_describe_supported_is_sorted_and_handles_empty():
    assert describe_supported() == "gre/aw, gre/quant, ielts/mock, ielts/reading, ielts/writing"
    assert describe_supported([]) == "none yet"
