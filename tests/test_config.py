from qpgen.config import load_config


def test_config_loads_all_sections():
    cfg = load_config()
    assert cfg.seed == 42
    assert set(cfg.subjects) == {"dbms", "os", "cn", "dsa"}
    assert cfg.marks == {"easy": 2, "medium": 5, "hard": 10}


def test_every_subject_has_six_to_eight_topics():
    cfg = load_config()
    for subject in cfg.subjects.values():
        assert 6 <= len(subject["topics"]) <= 8


def test_presets_add_up_to_total_marks():
    cfg = load_config()
    for counts in cfg.paper["presets"].values():
        total = sum(cfg.marks[level] * n for level, n in counts.items())
        assert total == cfg.paper["total_marks"]


def test_path_is_absolute():
    assert load_config().path("raw_dir").is_absolute()
