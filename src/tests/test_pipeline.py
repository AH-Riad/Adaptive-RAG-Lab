from src.core.adaptive_context import AdaptiveContext
from src.core.component import Component
from src.core.pipeline import AdaptivePipeline


class DummyComponent(Component):

    def run(self, context):

        context.add_event("dummy_executed")

        context.metadata["status"] = "success"

        return context


def main():

    context = AdaptiveContext(
        query="Explain self-attention"
    )

    pipeline = AdaptivePipeline()

    pipeline.add(DummyComponent())

    context = pipeline.run(context)

    print("=" * 60)

    print("Query")
    print(context.query)

    print("=" * 60)

    print("Events")
    print(context.events)

    print("=" * 60)

    print("Metadata")
    print(context.metadata)

    print("=" * 60)

    print("Logs")
    print(context.logs)


if __name__ == "__main__":
    main()