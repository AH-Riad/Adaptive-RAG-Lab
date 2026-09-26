from src.tests.test_topk_integrity import main as existing_topk_main


def test_existing_topk_integrity_gate():
    existing_topk_main()


if __name__ == "__main__":
    test_existing_topk_integrity_gate()
