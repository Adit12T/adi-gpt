from adigpt.db import fts_query
from adigpt.memory import explicit_fact, looks_secret
from adigpt.prompts import system_prompt
from adigpt.rag import MAX_CHARS, chunk_text, format_sources, verify_citations
from adigpt.tools import calculator, current_datetime, run_tool


def test_calculator_arithmetic():
    assert calculator("17 * 23") == "391"
    assert calculator("(1200 * 1.08) / 12") == "108"
    assert calculator("2 ^ 10") == "1024"
    assert calculator("sqrt(16) + abs(-2)") == "6"
    assert calculator("1 / 0") == "error: division by zero"


def test_calculator_rejects_code():
    for bad in ("__import__('os').system('id')", "open('/etc/passwd').read()",
                "(1).__class__", "[x for x in range(3)]", "9 ** 9 ** 9", "'a' * 5"):
        assert calculator(bad).startswith("error"), bad


def test_tools_dispatch():
    assert run_tool("calculator", {"expression": "2+2"}) == "4"
    assert run_tool("nope", {}).startswith("error")
    assert current_datetime()[:3] in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def test_fts_query_is_safe():
    assert fts_query('what is "NEAR( the retry* policy') == '"near" OR "retry" OR "policy"'
    assert fts_query("!!! ??? ") == ""


def test_chunk_markdown_keeps_heading_trail():
    text = "# Guide\nintro\n\n## Setup\nstep one\n\n### Linux\napt install\n\n## Usage\nrun it\n"
    chunks = chunk_text(text, "guide.md")
    assert ("Guide > Setup > Linux", "apt install") in chunks
    assert ("Guide > Usage", "run it") in chunks


def test_chunk_ignores_hashes_inside_code_fences():
    text = "# Title\n```sh\n# not a heading\nls\n```\n"
    assert chunk_text(text, "a.md") == [("Title", "```sh\n# not a heading\nls\n```")]


def test_chunk_size_limit_and_no_loss():
    words = " ".join(f"word{i}" for i in range(3000))
    chunks = chunk_text(words + "\n\n" + "x" * 5000, "big.txt")
    assert all(len(c) <= MAX_CHARS for _, c in chunks)
    joined = " ".join(c for _, c in chunks)
    assert "word0 " in joined and "word2999" in joined
    assert sum(c.count("x") for _, c in chunks) == 5000
    assert chunk_text("   \n\n ", "empty.txt") == []


def test_verify_citations():
    sources = [{"title": "a", "text": "The billing service listens on port 8431 in production."},
               {"title": "b", "text": "Rollback uses the rollback script with a release id."}]
    good = verify_citations("The billing service listens on port 8431 in production [S1].", sources)
    assert good["ok"] and good["cited"] == [1]
    fake = verify_citations("The billing service listens on port 8431 in production [S7].", sources)
    assert fake["invalid"] == [7] and not fake["ok"]
    bare = verify_citations("The billing service listens on port 8431 in production today.", sources)
    assert bare["uncited"] == 1 and not bare["ok"]
    weak = verify_citations("Giraffes migrate across frozen tundra every single winter season [S2].",
                            sources)
    assert weak["weak"] == [2] and not weak["ok"]
    assert verify_citations("NOT_IN_SOURCES\nNothing about that.", sources)["abstained"]


def test_format_sources_numbers_from_one():
    out = format_sources([{"title": "T", "heading": "H", "text": "body"},
                          {"title": "W", "url": "https://x.test", "text": "page"}])
    assert "[S1] T — H\nbody" in out and "[S2] W (https://x.test)\npage" in out


def test_memory_helpers():
    assert explicit_fact("Remember that I prefer Java 21") == "I prefer Java 21"
    assert explicit_fact("remember: standup is at 9:45") == "standup is at 9:45"
    assert explicit_fact("Do you remember the Alamo?") is None
    assert looks_secret("my api key is abc") and looks_secret("sk-abcdefghijklmnopqrstuv")
    assert looks_secret("card 4111 1111 1111 1111")
    assert not looks_secret("I lead a team of four developers")


def test_system_prompt_is_stable():
    a = system_prompt("spring", ["likes tea"], "be brief")
    assert a == system_prompt("spring", ["likes tea"], "be brief")
    assert "likes tea" in a and "be brief" in a and "Spring Boot architect" in a
    assert "Mode:" not in system_prompt("general", [])
