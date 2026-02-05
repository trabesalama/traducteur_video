from prompt_templates.translation_checklist import check_subtitles_preserve_timestamps

ORIG = """1
00:00:00,000 --> 00:00:02,000
Hello.

2
00:00:02,500 --> 00:00:05,000
How are you?
"""

TRANS_GOOD = """1
00:00:00,000 --> 00:00:02,000
Bonjour.

2
00:00:02,500 --> 00:00:05,000
Comment ça va?
"""

TRANS_BAD_TS = """1
00:00:00,100 --> 00:00:02,000
Bonjour.

2
00:00:02,500 --> 00:00:05,000
Comment ça va?
"""

TRANS_BAD_BLOCKS = """1
00:00:00,000 --> 00:00:02,000
Bonjour.
"""


def test_preserved_simple():
    assert check_subtitles_preserve_timestamps(ORIG, TRANS_GOOD)


def test_modified_timestamp_detected():
    assert not check_subtitles_preserve_timestamps(ORIG, TRANS_BAD_TS)


def test_different_block_count_detected():
    assert not check_subtitles_preserve_timestamps(ORIG, TRANS_BAD_BLOCKS)
