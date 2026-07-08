//! Adversarial session against the tier-0 scanner: prose that carries
//! digits and operator-shaped characters but asks for no computation.
use larql_inference::experts::arith::extract::find_expression;

fn main() {
    let cases: &[&str] = &[
        // spaced ranges / scores / idioms — the '-' with whitespace family
        "It takes 5 - 10 business days.",
        "They won 3 - 1 at home.",
        "I work a 9 - 5 job.",
        "Open Monday - Friday, 9 - 17.",
        "pages 12 - 48 cover the appendix",
        "the score was 2 - 2 after extra time",
        "ages 18 - 25 only",
        "dated 2026 - 06 - 11 in the ledger",
        // 'x' family
        "a 4 x 4 truck",
        "2 x 4 lumber at the yard",
        "a 3 x 5 index card",
        "room is 12 x 14 feet",
        // '+' in prose
        "I have 2 + years of experience",
        "rated 4 + stars on average",
        "C++ 11 added move semantics",
        "call +44 7911 123456",
        "she scored 1600 + on the test",
        // metaphor words (MEE territory — must be inert in AVE v0.1)
        "exponential growth of 300 users",
        "let me go off on a tangent about 7 things",
        "check the log file at line 42",
        "a sine of the times, all 9 of them",
        // ambiguous bare/question weak forms — now the model's territory
        "9 - 5",
        "what is 100 - 7?",
        "Are you available 9 - 5?",
        // legit math notation that MUST keep firing
        "12 + 7 =",
        "what is 123456 + 654321?",
        "100000 - 1 =",
        "12345 * 6789",
        "3 x 4 =",
        "47−5",
        "999 + 111 - 222 =",
    ];
    for c in cases {
        match find_expression(c) {
            Some(e) => println!("FIRE   {c:<46} -> {} = {}", e, e.eval()),
            None => println!("  no   {c}"),
        }
    }
}
