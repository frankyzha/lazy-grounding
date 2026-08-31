from __future__ import annotations

from lazy_grounding.schemas import Label, Rewrite
from lazy_grounding.scoring import OpenAIAnswerJudge, ScoreRequest, score_answer


def request(response: str, *, surfaced: bool = True) -> ScoreRequest:
    return ScoreRequest(
        original_question="Who won in 2024?",
        original_answer="Alice",
        rewrites=(Rewrite(question="Who won in 2023?", answer="Bob"),),
        model_response=response,
        nearby_evidence_surfaced=surfaced,
    )


def unused_judge(_: ScoreRequest) -> dict[str, object]:
    raise AssertionError("Exact matching should not call the judge")


def test_exact_original_and_nearby_answers_do_not_call_judge() -> None:
    correct = score_answer(request("<answer>Alice</answer>"), judge=unused_judge)
    adopted = score_answer(request("The result is <answer>Bob.</answer>"), judge=unused_judge)
    assert correct.label is Label.CORRECT
    assert adopted.label is Label.CR_R
    assert adopted.matched_rewrite_index == 0


def test_nearby_match_without_exposure_is_cr_p() -> None:
    result = score_answer(request("Bob", surfaced=False), judge=unused_judge)
    assert result.label is Label.CR_P


def test_semantic_judge_maps_other_wrong_by_exposure() -> None:
    def judge(_: ScoreRequest) -> dict[str, object]:
        return {
            "best_label": "OTHER_WRONG",
            "final_answer": "Carol",
            "matched_rewrite_index": None,
            "rationale": "Matches neither answer.",
        }

    exposed = score_answer(request("The likely winner was Carol."), judge=judge)
    unexposed = score_answer(request("The likely winner was Carol.", surfaced=False), judge=judge)
    assert exposed.label is Label.CA_OW
    assert unexposed.label is Label.NC_W


def test_semantic_nearby_label_requires_valid_index() -> None:
    def judge(_: ScoreRequest) -> dict[str, object]:
        return {
            "best_label": "NEARBY_ANSWER",
            "final_answer": "Robert",
            "matched_rewrite_index": 2,
            "rationale": "Alias of Bob.",
        }

    try:
        score_answer(request("Robert"), judge=judge)
    except RuntimeError as error:
        assert "out-of-range" in str(error)
    else:
        raise AssertionError("Invalid rewrite index should fail")


def test_openai_judge_uses_structured_response() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {
                "output_text": (
                    '{"final_answer":"Robert","best_label":"NEARBY_ANSWER",'
                    '"matched_rewrite_index":0,"rationale":"Alias."}'
                )
            }

    def post(*args: object, **kwargs: object) -> Response:
        return Response()

    judge = OpenAIAnswerJudge(api_key="key", model="gpt-5.4", post=post)
    result = score_answer(request("Robert"), judge=judge)
    assert result.label is Label.CR_R
    assert result.judge_model == "gpt-5.4"
