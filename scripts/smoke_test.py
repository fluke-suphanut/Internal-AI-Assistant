from __future__ import annotations

import json
import sys

from app.agent.agent import AIAgent
from app.core.logging import setup_logging

logger = setup_logging()


def main():
    agent = AIAgent()

    print("\n=== Smoke Test: Internal Q&A ===")
    qa_query = "What issues were reported about email notifications?"

    qa_response = agent.run(
        user_text=qa_query,
        top_k=3,
    )

    print(json.dumps(qa_response.model_dump(), indent=2, ensure_ascii=False))

    print("\n=== Smoke Test: Issue Summary ===")
    issue_text = (
        "Users report that email notifications are delayed by several hours. "
        "Some users say notifications are not sent at all during peak traffic."
    )

    summary_response = agent.run_issue_summary(
        issue_text=issue_text,
    )

    print(json.dumps(summary_response.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
        print("\nSmoke test completed successfully")
    except Exception as e:
        print("\nSmoke test failed")
        print(str(e))
        sys.exit(1)
