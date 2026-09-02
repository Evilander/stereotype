from stereotype_audit import pairs


def test_fill_handles_capitalised_slot():
    assert pairs.fill("{Cue} works as {attr}.", cue="a woman", attr="a nurse") == "A woman works as a nurse."
    assert pairs.fill("{cue} is here", cue="Emily") == "Emily is here"


def test_make_pairs_expands_and_is_deterministic():
    tpls = [{"id": "t1", "text": "{Cue} is here."}, {"id": "t2", "text": "I met {cue}."}]
    groups = {"g1": ["a man", "a boy"], "g2": ["a woman"]}
    out = pairs.make_pairs("gender", tpls, groups, [("g1", "g2")], seed=0)
    assert len(out) == 4
    assert out[0].left == "A man is here." and out[0].right == "A woman is here."
    assert out[0].group_left == "g1" and out[0].group_right == "g2"
    again = pairs.make_pairs("gender", tpls, groups, [("g1", "g2")], seed=0)
    assert [p.pair_id for p in out] == [p.pair_id for p in again]
    sub = pairs.make_pairs("gender", tpls, groups, [("g1", "g2")], seed=0, max_pairs=2)
    assert len(sub) == 2


def test_shuffle_cues_swaps_about_half_and_keeps_labels_attached():
    tpls = [{"id": "t1", "text": "{Cue} is here."}]
    groups = {"g1": [f"n{i}" for i in range(50)], "g2": ["x"]}
    out = pairs.make_pairs("axis", tpls, groups, [("g1", "g2")])
    shuffled = pairs.shuffle_cues(out, seed=0)
    swapped = sum(1 for p in shuffled if p.group_left == "g2")
    assert 10 < swapped < 40
    for p in shuffled:
        if p.group_left == "g2":
            assert p.cue_left == "x" and p.left == "X is here."


def test_packaged_cues_and_templates_load():
    names = pairs.load_cues("names_bm2004")
    assert len(names["groups"]) == 4 and all(len(v) == 9 for v in names["groups"].values())
    iat = pairs.load_cues("iat_attributes")
    assert len(iat["sets"]["pleasant"]) == 25 and len(iat["sets"]["unpleasant"]) == 25
    groups = pairs.load_cues("groups")
    assert "religion" in groups["axes"]
    for name in ["assoc", "nameswap", "chatiat", "seat", "retrieval", "ctf"]:
        assert pairs.load_templates(name)["battery"] == name
