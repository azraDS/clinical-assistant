from clinical_assistant.kdigo_lookups import RISK_HEATMAP, acr_category, gfr_category


def test_gfr_category_boundaries():
    assert gfr_category(94) == "G1"
    assert gfr_category(48) == "G3a"
    assert gfr_category(38) == "G3b"
    assert gfr_category(22) == "G4"
    assert gfr_category(8) == "G5"
    assert gfr_category(None) is None


def test_acr_category_boundaries():
    assert acr_category(10) == "A1"
    assert acr_category(250) == "A2"
    assert acr_category(850) == "A3"
    assert acr_category(None) is None


def test_risk_heatmap_covers_every_category():
    for g in ("G1", "G2", "G3a", "G3b", "G4", "G5"):
        for a in ("A1", "A2", "A3"):
            assert RISK_HEATMAP[g][a] in ("low", "moderate", "high", "very high")
