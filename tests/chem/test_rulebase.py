from retrochimera.chem.rules import RuleBase

rules = {
    645: "([#6A&h0X3v4+0:20]-[#7A&h1X3v3+0:22])>>([#6A&h0X3v4+0:20]-[#8A&h1X2v2+0:38].[#7A&h2X3v3+0:22])",
    984: "([#8A&h1X2v2+0:1])>>([#8A&h0X2v2+0:1]-[#6A&h3X4v4+0:16])",
    319: "([#7A&h2X3v3+0:21])>>([#7A&h0X3v4+:21](=[#8A&h0X&v2+0:25])-[#8A&h0X&v-:26])",
    630: "([#8A&h0X2v2+0:10]-[#6A&h2X4v4+0:11])>>([#6A&h2X4v4+0:11]-[BrA&h0X&v+0:19].[#8A&h1X2v2+0:10])",
}


def test_rule_base() -> None:

    rulz = RuleBase()

    for k, v in rules.items():
        rulz.add_rule(v, rule_id=k, rule_hash=f"#{k}")

    assert (
        rulz[645].smarts
        == "([#6A&h0X3v4+0:20]-[#7A&h1X3v3+0:22])>>([#6A&h0X3v4+0:20]-[#8A&h1X2v2+0:38].[#7A&h2X3v3+0:22])"
    )

    assert rulz[984].smarts == "([#8A&h1X2v2+0:1])>>([#8A&h0X2v2+0:1]-[#6A&h3X4v4+0:16])"

    rule = rulz.get_rule_from_hash("#630")
    assert rule is not None
    assert (
        rule.smarts
        == "([#8A&h0X2v2+0:10]-[#6A&h2X4v4+0:11])>>([#6A&h2X4v4+0:11]-[BrA&h0X&v+0:19].[#8A&h1X2v2+0:10])"
    )

    assert 319 in rulz

    assert 666 not in rulz


def test_rulebase_io() -> None:

    rulz = RuleBase()

    for k, v in rules.items():
        rulz.add_rule(v, rule_id=k, rule_hash=f"#{k}")


def test_rule_base_no_ids() -> None:
    rulz = RuleBase()

    for k, v in rules.items():
        rulz.add_rule(v, rule_id=None, rule_hash=None)

    assert rulz[1].smarts in rules.values()
