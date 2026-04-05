from llm_chunker import LLMChunker, QwenClient


class SampleText:
    """Example texts for testing the chunker."""

    MIXED_TOPICS = """
    Machine learning is a subset of artificial intelligence.
    It allows systems to learn from data without explicit programming.
    Common algorithms include decision trees, neural networks, and SVMs.

    The Eiffel Tower was built in 1889 in Paris for the World's Fair.
    It stands 330 meters tall and attracts millions of tourists each year.
    Gustave Eiffel designed the iron lattice structure.

    Photosynthesis is the process by which plants convert sunlight into energy.
    Chlorophyll in the leaves absorbs light, while CO2 is taken from the air.
    Oxygen is released as a byproduct of this reaction.
    """


if __name__ == "__main__":
    chunker = LLMChunker(client=QwenClient(thinking=False))
    chunks = chunker.chunk(SampleText.MIXED_TOPICS)

    print(f"\n{len(chunks)} Chunks gefunden:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk)
        print()
