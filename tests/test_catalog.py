from shared.src.shared.catalog import SUPPORTED_PAIRS, describe_supported, is_supported


def test_chat_supports_gre_quant_and_aw_and_the_ielts_writing_coach():
    assert set(SUPPORTED_PAIRS) == {("gre", "quant"), ("gre", "aw"), ("ielts", "writing")}
    assert is_supported("gre", "quant") and is_supported("gre", "aw")
    assert is_supported("ielts", "writing")
    assert not is_supported("gre", "verbal")
    assert not is_supported("ielts", "speaking")


def test_describe_supported_is_sorted_and_handles_empty():
    assert describe_supported() == "gre/aw, gre/quant, ielts/writing"
    assert describe_supported([]) == "none yet"
